from math import dist
import numpy as np
from torch import layout, mode
from src.main import geometry_operations
import plotly.graph_objects as go


"""For Radar Chart"""

import pandas as pd
from plotly.colors import hex_to_rgb
import plotly.express as px
from typing import List, Optional, Tuple, Dict, Any


def _scale(series: pd.Series, pad_low: float = 0.1) -> pd.Series:
    """Min-max to [0, 1], but the min is lowered by pad_low * (max - min)"""

    s = pd.to_numeric(series, errors="coerce")
    min, max = s.min(skipna=True), s.max(skipna=True)
    rng = max - min

    if pd.isna(rng) or rng == 0:
        return pd.Series(0.5, index=s.index, dtype="float64")
    
    min_adj = min - pad_low * rng
    return ((s - min_adj) / (max - min_adj)).clip(0, 1)


def _convert_hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _triangle_area_on_axes(row: pd.Series, axes: List[str]) -> float:
    """
    Calculate the area of the triangle formed by the points on the given axes.
    Shoelace formula in Cartesian after projecting.
    """

    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    r = np.array([row[a] for a in axes], dtype=float)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[0] - y[0]*x[1] - y[1]*x[2] - y[2]*x[0])

def make_radar_scores(results_df: pd.DataFrame, axes: List[str])-> pd.DataFrame:
    df = results_df.copy()

    eco = _scale(df["ecological_distances_RNI"])
    ergo = _scale(df["ergonomics_distances_RNI"])
    cost = _scale(df["cost_objective_RNI"])

    scores = pd.DataFrame({
        "Name": [f"{i+1}" for i in df.index],
        "Ökologische Optimierung": eco,
        "Ergonomische Optimierung": ergo,
        "Kosten Optimierung": cost,
    }, index=df.index)

    colors = [px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, _ in enumerate(scores.index)]
    scores["color"]= [ _convert_hex_to_rgba(color) for color in colors]
    scores["fill_color"]= [ _convert_hex_to_rgba(color, 0.18) for color in colors]
    scores["raw_eco"] = results_df.loc[df.index, "ecological_distances_RNI"]
    scores["raw_ergo"] = results_df.loc[df.index, "ergonomics_distances_RNI"]
    scores["raw_cost"] = results_df.loc[df.index, "cost_objective_RNI"]

    scores["triangle_area"] = scores.apply(_triangle_area_on_axes, axis=1, args=(axes,))

    return scores

""" End of radar chart """






def update_layout_overview(indices, forest_area_3, model_list, precomputed=None) -> dict:
    """
    Compute metrics for the given selection of cable corridors (real indices).
    Uses optional precomputed distance matrices for speed.
    """
    rot_line_gdf = forest_area_3.line_gdf[forest_area_3.line_gdf.index.isin(indices)]

    # Precompute distances once and slice -> big speedup
    if precomputed is not None and len(indices) > 0:
        full_idx = forest_area_3.line_gdf.index
        cols = [int(np.where(full_idx == i)[0][0]) for i in indices]
        distance_tree_line = precomputed[0][:, cols]
        distance_carriage_support = precomputed[1][:, cols]
    else:
        (
            distance_tree_line,
            distance_carriage_support,
        ) = geometry_operations.compute_distances_facilities_clients(
            forest_area_3.harvesteable_trees_gdf, rot_line_gdf
        )

    # assign all trees to their closest line
    try:
        tree_to_line_assignment = np.argmin(distance_tree_line, axis=1)

        # compute the distance of each tree to its assigned line
        distance_trees_to_selected_lines = distance_tree_line[
            range(len(tree_to_line_assignment)), tree_to_line_assignment
        ]
    except Exception:
        tree_to_line_assignment = [0 for _ in range(len(forest_area_3.harvesteable_trees_gdf))]
        distance_trees_to_selected_lines = []

    # compute the productivity cost
    if len(indices) > 0:
        selected_prod_cost = model_list[0].productivity_cost[:, indices]
    else:
        selected_prod_cost = np.zeros_like(model_list[0].productivity_cost[:, :1])

    productivity_cost_overall = 0
    for index, val in enumerate(tree_to_line_assignment):
        val = min(val, selected_prod_cost.shape[1] - 1)  # guard
        productivity_cost_overall += selected_prod_cost[index][val]

    # sum of wood volume per CR
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(max(1, len(rot_line_gdf)))
    ]
    wood_volume_per_cr = [
        int(
            sum(
                forest_area_3.harvesteable_trees_gdf.iloc[grouped_indices]["cubic_volume"]
            )
        )
        for grouped_indices in grouped_class_indices
    ][: len(rot_line_gdf)]

    # average tree size per CR
    average_tree_size_per_cr = [
        round(
            sum(forest_area_3.harvesteable_trees_gdf.iloc[grouped_indices]["h"]) / len(grouped_indices),
            2,
        )
        if len(grouped_indices) > 0 else 0
        for grouped_indices in grouped_class_indices
    ][: len(rot_line_gdf)]

    # supports
    supports_height = [
        (
            [
                segment.start_support.attachment_height
                for segment in cr_object.supported_segments[1:]
            ]
            if cr_object.supported_segments
            else []
        )
        for cr_object in rot_line_gdf["Cable Road Object"]
    ]
    supports_amount = [len(heights) for heights in supports_height]

    # yarding distances per CR
    max_yarding_distance_per_cr = []
    average_yarding_distance_per_cr = []
    for line_idx, grouped_indices in enumerate(grouped_class_indices[: len(rot_line_gdf)]):
        if len(grouped_indices) == 0 or len(indices) == 0:
            max_yarding_distance_per_cr.append(0)
            average_yarding_distance_per_cr.append(0)
        else:
            dists = distance_carriage_support[grouped_indices, line_idx]
            max_yarding_distance_per_cr.append(int(max(dists)))
            average_yarding_distance_per_cr.append(int(np.mean(dists)))

    # tail spar (end mast)
    endmast_height_list = []
    endmast_BHD_list = []
    endmast_max_holding_force_list = []
    endmast_x_list = []
    endmast_y_list = []
    for end_support_tree in rot_line_gdf.end_support_tree:
        endmast_height_list.append(int(end_support_tree["h"]))
        endmast_BHD_list.append(int(end_support_tree["BHD"]))
        endmast_max_holding_force_list.append(
            int(end_support_tree["max_holding_force"])
        )
        endmast_x_list.append(round(end_support_tree["x"], 2))
        endmast_y_list.append(round(end_support_tree["y"], 2))

    # road anchor
    road_anchor_height_list = []
    road_anchor_BHD_list = []
    road_anchor_max_holding_force_list = []
    road_anchor_x_list = []
    road_anchor_y_list = []
    for end_support_tree in rot_line_gdf.end_support_tree:
        endmast_height_list.append(int(end_support_tree["h"]))
        endmast_BHD_list.append(int(end_support_tree["BHD"]))
        endmast_max_holding_force_list.append(
            int(end_support_tree["max_holding_force"])
        )
        endmast_x_list.append(round(end_support_tree["x"], 2))
        endmast_y_list.append(round(end_support_tree["y"], 2))

    # global yarding distances
    max_yarding_distance = int(max(distance_trees_to_selected_lines)) if len(distance_trees_to_selected_lines) else 0
    average_yarding_distance = int(np.mean(distance_trees_to_selected_lines)) if len(distance_trees_to_selected_lines) else 0

    line_cost = int(sum(rot_line_gdf["line_cost"])) if len(rot_line_gdf) else 0

    # total cost
    total_cable_road_costs = int(line_cost + productivity_cost_overall)

    cost_per_m3 = round(total_cable_road_costs / max(1, sum(wood_volume_per_cr) if len(wood_volume_per_cr) else 1), 2)

    # ecological penalty
    if len(indices) > 0:
        ecological_penalty_threshold = 10
        ecological_penalty_lateral_distances = np.where(
            distance_tree_line > ecological_penalty_threshold,
            distance_tree_line - ecological_penalty_threshold,
            0,
        )
        sum_eco_distances = int(
            sum(
                [
                    ecological_penalty_lateral_distances[j][i]
                    for i, j in zip(
                        tree_to_line_assignment,
                        range(len(ecological_penalty_lateral_distances)),
                    )
                ]
            )
        )
    else:
        sum_eco_distances = 0

    # ergonomics penalty (double beyond threshold)
    if len(indices) > 0:
        ergonomics_penalty_treshold = 15
        ergonomic_penalty_lateral_distances = np.where(
            distance_tree_line > ergonomics_penalty_treshold,
            (distance_tree_line - ergonomics_penalty_treshold) * 2,
            0,
        )
        sum_ergo_distances = int(
            sum(
                [
                    ergonomic_penalty_lateral_distances[j][i]
                    for i, j in zip(
                        tree_to_line_assignment,
                        range(len(ergonomic_penalty_lateral_distances)),
                    )
                ]
            )
        )
    else:
        sum_ergo_distances = 0

    # volume per running meter
    total_cable_road_length = float(sum(rot_line_gdf["line_length"])) if len(rot_line_gdf) else 0.0
    volume_per_running_meter = round((sum(wood_volume_per_cr) / total_cable_road_length) if total_cable_road_length else 0.0, 2)

    return {
        "Wood Volume per Cable Corridor (m3)": wood_volume_per_cr,
        "Total Cable Corridor Costs (€)": total_cable_road_costs,
        "Setup and Takedown, Prod. Costs (€)": f"{line_cost} / {int(productivity_cost_overall)}",
        "Ecol. Penalty": sum_eco_distances,
        "Ergon. Penalty": sum_ergo_distances,
        "Tree to Cable Corridor Assignment": tree_to_line_assignment if len(indices) > 0 else [0]*len(forest_area_3.harvesteable_trees_gdf),
        "Supports Height (m)": supports_height,
        "Supports Amount": supports_amount,
        "Max lateral Yarding Distance (m)": max_yarding_distance,
        "Average lateral Yarding Distance (m)": average_yarding_distance,
        "Cost per m3 (€)": cost_per_m3,
        "Average Tree Height (m)": average_tree_size_per_cr,
        "Volume per Meter (m3/m)": volume_per_running_meter,
        "Max Yarding Distance per Cable Corridor (m)": max_yarding_distance_per_cr,
        "Average Yarding Distance per Cable Corridor (m)": average_yarding_distance_per_cr,
        "Anchor height": endmast_height_list,
        "Anchor BHD": endmast_BHD_list,
        "Anchor max holding force": endmast_max_holding_force_list,
        "Anchor x coordinate": endmast_x_list,
        "Anchor y coordinate": endmast_y_list,
        "Corresponding Cable Corridor": indices,  # real indices
        "Road Anchor height": road_anchor_height_list,
        "Road Anchor BHD": road_anchor_BHD_list,
        "Road Anchor max holding force": road_anchor_max_holding_force_list,
        "Road Anchor x coordinate": road_anchor_x_list,
        "Road Anchor y coordinate": road_anchor_y_list,
        "Road Anchor Angle of Attack": rot_line_gdf["angle_between_start_support_and_cr"] if len(rot_line_gdf) else [],
        "Tail Anchor Angle of Attack": rot_line_gdf["angle_between_end_support_and_cr"] if len(rot_line_gdf) else [],
    }


def get_overview_table_data(forest_area_3, model_list, results_df) -> List[List[str]]:

    all_distance_tree_line, all_distance_carriage_support = geometry_operations.compute_distances_facilities_clients(
        forest_area_3.harvesteable_trees_gdf, forest_area_3.line_gdf
    )
    indices_to_show = sorted({int(i) for row in results_df["selected_lines"] for i in row})
    display_names = list(range(1, len(indices_to_show) + 1))

    real_to_display = dict(zip(indices_to_show, display_names))

    precomputed = (all_distance_tree_line, all_distance_carriage_support)

    results = []

    for i, row in results_df.iterrows():
        sel = list(map(int, row["selected_lines"]))
        layout_data = update_layout_overview(sel, forest_area_3, model_list, precomputed=precomputed)

        results.append(
            [
                i + 1,
                layout_data["Total Cable Corridor Costs (€)"],
                layout_data["Setup and Takedown, Prod. Costs (€)"],
                layout_data["Ecol. Penalty"],
                layout_data["Ergon. Penalty"],
                [real_to_display.get(int(idx), int(idx)) for idx in sel],
                layout_data["Max lateral Yarding Distance (m)"],
                layout_data["Average lateral Yarding Distance (m)"],
                int(np.mean(layout_data["Supports Amount"])) if layout_data.get("Supports Amount") else 0,
                layout_data["Cost per m3 (€)"],
                layout_data["Volume per Meter (m3/m)"],
            ]
        )

    return results


def _get_selected_lines(results_df: pd.DataFrame, selected_index: Optional[int]) -> List[int]:
    if selected_index is None:
        return []
    if selected_index < 0 or selected_index >= len(results_df):
        return []
    
    return list(map(int, results_df.iloc[int(selected_index)]["selected_lines"]))

def get_selected_table_data(forest_area_3, model_list, results_df: pd.DataFrame, selected_index: Optional[int]) -> List[List[str]]:
    selected_lines = _get_selected_lines(results_df, selected_index)
    if not selected_lines:
        return []
    
    layout_data = update_layout_overview(selected_lines, forest_area_3, model_list, precomputed=None)

    volumes = layout_data["Wood Volume per Cable Corridor (m3)"]
    supports_amount = layout_data["Supports Amount"]
    supports_height = layout_data["Supports Height (m)"]
    avg_tree_h = layout_data["Average Tree Height (m)"]
    max_yarding = layout_data["Max Yarding Distance per Cable Corridor (m)"]
    avg_yarding = layout_data["Average Yarding Distance per Cable Corridor (m)"]

    subset = forest_area_3.line_gdf.loc[forest_area_3.line_gdf.index.isin(selected_lines)]
    subset = subset.loc[selected_lines]

    rows: List[List[str]] = []

    for i, real_idx in enumerate(selected_lines):
        line_cost = int(subset.loc[real_idx, "line_cost"]) if "line_cost" in subset.columns else 0
        line_length = int(subset.loc[real_idx, "line_length"]) if "line_length" in subset.columns else 0

        height_list = supports_height[i] if i < len(supports_height) else []
        heights_str = "/" if not height_list else ", ".join(str(int(h)) for h in height_list)

        avg_h = float(avg_tree_h[i]) if i < len(avg_tree_h) else 0.0
        max_y = int(max_yarding[i]) if i < len(max_yarding) else 0
        avg_y = int(avg_yarding[i]) if i < len(avg_yarding) else 0
        vol = int(volumes[i]) if i < len(volumes) else 0
        sup_n = int(supports_amount[i]) if i < len(supports_amount) else 0

        rows.append([
            str(real_idx + 1),
            str(line_cost),
            str(line_length),
            str(vol),
            str(sup_n),
            heights_str,
            f"{avg_h:.2f}",
            str(max_y),
            str(avg_y),
        ])

    return rows

def get_anchor_table_data(forest_area_3, model_list, results_df, selected_index: Optional[int]) -> List[List[str]]:
    selected_lines = _get_selected_lines(results_df, selected_index)
    if not selected_lines:
        return []
    
    layout_data = update_layout_overview(selected_lines, forest_area_3, model_list, precomputed=None)

    corr = layout_data["Corresponding Cable Corridor"]
    bhd = layout_data["Anchor BHD"]
    h = layout_data["Anchor height"]
    xcoord = layout_data["Anchor x coordinate"]
    ycoord = layout_data["Anchor y coordinate"]

    n = min(len(corr), len(bhd), len(h), len(xcoord), len(ycoord))
    rows: List[List[str]] = []
    for i in range(n):
        rows.append([
            str(int(corr[i] if i < len(corr) else -1)),
            str(int(bhd[i] if i < len(bhd) else 0)),
            str(int(h[i] if i < len(h) else 0)),
            f"{float(xcoord[i]):.2f}" if i < len(xcoord) else "0.00",
            f"{float(ycoord[i]):.2f}" if i < len(ycoord) else "0.00",
        ])

    return rows

def _sample_line_xy(line, min_points: int = 20, step: float = 5.0) -> Tuple[List[float], List[float]]:
    length = float(line.length)
    n_points = max(int(length // step) + 2, min_points)
    dists = np.linspace(0.0, length, n_points)
    xs, ys = [], []
    for d in dists:
        p = line.interpolate(d)
        xs.append(float(p.x))
        ys.append(float(p.y))
    return xs, ys

def prepare_map_data(forest_area_3, results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Precompute trees, corridors, anchors, and display/index maps.
    • Only include corridors that appear in at least one optimization result.
    • Tail anchor coords are the line end-point (end tree of the corridor).
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # --- ONLY corridors present in any optimization result (old interface does this) ---
    # indices_to_show will be REAL indices (line_gdf index)
    indices_to_show = sorted({int(i) for row in results_df["selected_lines"] for i in row})
    display_names   = list(range(1, len(indices_to_show) + 1))
    display_to_real = dict(zip(display_names, indices_to_show))
    real_to_display = dict(zip(indices_to_show, display_names))
    # (Old behavior reference) :contentReference[oaicite:1]{index=1}

    # Stable base palette (used for default/neutral view; selection uses per-selection colors)
    palette  = px.colors.qualitative.Plotly
    color_map = {idx: palette[i % len(palette)] for i, idx in enumerate(indices_to_show)}

    # Trees
    tree_x, tree_y, tree_bhd = [], [], []
    for geom, bhd in zip(
        forest_area_3.harvesteable_trees_gdf.geometry,
        forest_area_3.harvesteable_trees_gdf.get("BHD", pd.Series([None]*len(forest_area_3.harvesteable_trees_gdf)))
    ):
        if hasattr(geom, "x") and hasattr(geom, "y"):
            x, y = float(geom.x), float(geom.y)
        else:
            x, y = geom.xy[0][0], geom.xy[1][0]
        tree_x.append(x)
        tree_y.append(y)
        tree_bhd.append(None if pd.isna(bhd) else int(bhd))

    # --- helper: dense-ish sampling along each polyline for nice hover everywhere ---
    def _sample_line_xy(line, min_points: int = 20, step: float = 5.0):
        length = float(line.length)
        n_points = max(int(length // step) + 2, min_points)
        dists = np.linspace(0.0, length, n_points)
        xs, ys = [], []
        for d in dists:
            p = line.interpolate(d)
            xs.append(float(p.x)); ys.append(float(p.y))
        return xs, ys

    # Corridors + anchors
    corridors: Dict[int, Dict[str, Any]] = {}
    for real_idx, row in forest_area_3.line_gdf.loc[indices_to_show].iterrows():
        line = row.geometry
        xs, ys = _sample_line_xy(line)
        start_pt = line.coords[0]
        end_pt   = line.coords[-1]

        # --- Tail anchor == end tree of the corridor: use the end-point coords ---
        # Keep BHD if available from end_support_tree, but coords are from `end_pt`
        tail_src = getattr(row, "end_anchor_tree", None)
        ex = float(end_pt[0]); ey = float(end_pt[1]); ebhd = 0  # fallback
        if isinstance(tail_src, dict):
            ex = float(tail_src.get("x", ex))
            ey = float(tail_src.get("y", ey))
            ebhd = int(tail_src.get("BHD", 0))
        else:
            # sometimes it's a pandas Series
            try:
                ex = float(tail_src.loc["x"])
                ey = float(tail_src.loc["y"])
                ebhd = int(tail_src.loc.get("BHD", 0))
            except Exception:
                pass  # keep fallback

        tail_anchor = dict(x=ex, y=ey, BHD=ebhd)

        # --- Road anchors: keep them all, normalized to list[dict] ---
        road_anchor_entries: List[dict] = []
        ra_src = getattr(row, "road_anchor_tree_series", None)
        if isinstance(ra_src, pd.DataFrame) and not ra_src.empty:
            for _, r in ra_src.iterrows():
                road_anchor_entries.append(
                    dict(x=float(r.get("x", 0)), y=float(r.get("y", 0)), BHD=int(r.get("BHD", 0)))
                )
        elif isinstance(ra_src, dict) and "features" in ra_src:
            for f in ra_src["features"]:
                props = f.get("properties", f)
                road_anchor_entries.append(
                    dict(x=float(props.get("x", 0)), y=float(props.get("y", 0)), BHD=int(props.get("BHD", 0)))
                )

        corridors[int(real_idx)] = dict(
            xs=xs, ys=ys,
            start=(float(start_pt[0]), float(start_pt[1])),
            end=(float(end_pt[0]), float(end_pt[1])),
            tail_anchor=tail_anchor,
            road_anchors=road_anchor_entries,
        )

    # Extents
    all_x = tree_x[:]; all_y = tree_y[:]
    for c in corridors.values():
        all_x += c["xs"]; all_y += c["ys"]
        all_x.append(c["tail_anchor"]["x"]); all_y.append(c["tail_anchor"]["y"])
        for ra in c["road_anchors"]:
            all_x.append(ra["x"]); all_y.append(ra["y"])
    x_range = (min(all_x) - 10, max(all_x) + 10) if all_x else None
    y_range = (min(all_y) - 10, max(all_y) + 10) if all_y else None

    return dict(
        tree_x=tree_x, tree_y=tree_y, tree_bhd=tree_bhd,
        corridors=corridors,
        indices_to_show=indices_to_show,
        display_to_real=display_to_real, real_to_display=real_to_display,
        color_map=color_map,               # base palette (neutral/legend use)
        x_range=x_range, y_range=y_range,
    )