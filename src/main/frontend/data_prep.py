# src/main/frontend/data_prep.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from src.main import geometry_operations

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.colors import hex_to_rgb


def _scale(series: pd.Series, pad_low: float = 0.1) -> pd.Series:
    """Min-max to [0, 1], but the min is lowered by pad_low * (max - min)."""
    s = pd.to_numeric(series, errors="coerce")
    minv, maxv = s.min(skipna=True), s.max(skipna=True)
    rng = maxv - minv
    if pd.isna(rng) or rng == 0:
        return pd.Series(0.5, index=s.index, dtype="float64")
    min_adj = minv - pad_low * rng
    return ((s - min_adj) / (maxv - min_adj)).clip(0, 1)


def _convert_hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _triangle_area_on_axes(row: pd.Series, axes: List[str]) -> float:
    """Area of the triangle formed by the points on given axes (radar triangle)."""
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    r = np.array([row[a] for a in axes], dtype=float)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[0] - y[0]*x[1] - y[1]*x[2] - y[2]*x[0])

def _compute_tree_labels_for_selection(selected_real_indices,
                                       full_line_index: np.ndarray,
                                       distance_tree_line: np.ndarray) -> Optional[np.ndarray]:
    """
    selected_real_indices: list[int] of REAL corridor indices (the same you pass around the UI)
    full_line_index: np.ndarray of all REAL indices in forest_area_3.line_gdf.index (column order of distance_tree_line)
    distance_tree_line: (n_trees x n_lines) full distance matrix

    Returns:
        labels: (n_trees,) int array with values in 0..k-1, where k = len(selected_real_indices),
                or None if selection is empty/invalid.
    """
    if not selected_real_indices:
        return None

    # map selection of REAL indices -> columns in the distance matrix
    cols = []
    for ridx in selected_real_indices:
        hit = np.where(full_line_index == int(ridx))[0]
        if hit.size:
            cols.append(int(hit[0]))
    if not cols:
        return None

    # argmin over the restricted columns gives per-tree label 0..k-1
    dsel = distance_tree_line[:, cols]
    return np.argmin(dsel, axis=1)

def update_layout_overview(indices, forest_area_3, model_list, precomputed=None) -> dict:
    """
    Compute metrics for the given selection of cable corridors (REAL indices).
    Uses optional precomputed distance matrices for speed.
    """
    rot_line_gdf = forest_area_3.line_gdf[forest_area_3.line_gdf.index.isin(indices)]

    # Precompute distances once and slice
    if precomputed is not None and len(indices) > 0:
        full_idx = forest_area_3.line_gdf.index
        cols = [int(np.where(full_idx == i)[0][0]) for i in indices]
        distance_tree_line = precomputed[0][:, cols]
        distance_carriage_support = precomputed[1][:, cols]
    else:
        distance_tree_line, distance_carriage_support = geometry_operations.compute_distances_facilities_clients(
            forest_area_3.harvesteable_trees_gdf, rot_line_gdf
        )

    # Assign trees to their closest selected line
    try:
        tree_to_line_assignment = np.argmin(distance_tree_line, axis=1)
        distance_trees_to_selected_lines = distance_tree_line[
            range(len(tree_to_line_assignment)), tree_to_line_assignment
        ]
    except Exception:
        tree_to_line_assignment = [0 for _ in range(len(forest_area_3.harvesteable_trees_gdf))]
        distance_trees_to_selected_lines = []

    # Productivity cost
    if len(indices) > 0:
        selected_prod_cost = model_list[0].productivity_cost[:, indices]
    else:
        selected_prod_cost = np.zeros_like(model_list[0].productivity_cost[:, :1])

    productivity_cost_overall = 0
    for index, val in enumerate(tree_to_line_assignment):
        val = min(val, selected_prod_cost.shape[1] - 1)  # guard
        productivity_cost_overall += selected_prod_cost[index][val]

    # Wood volume per corridor (sum of trees assigned to each corridor)
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(max(1, len(rot_line_gdf)))
    ]
    wood_volume_per_cr = [
        int(sum(forest_area_3.harvesteable_trees_gdf.iloc[g]["cubic_volume"]))
        for g in grouped_class_indices
    ][: len(rot_line_gdf)]

    # Average tree height per corridor
    average_tree_size_per_cr = [
        round(
            sum(forest_area_3.harvesteable_trees_gdf.iloc[g]["h"]) / len(g), 2
        ) if len(g) > 0 else 0
        for g in grouped_class_indices
    ][: len(rot_line_gdf)]

    # Supports
    supports_height = [
        (
            [segment.start_support.attachment_height for segment in cr_object.supported_segments[1:]]
            if cr_object.supported_segments else []
        )
        for cr_object in rot_line_gdf["Cable Road Object"]
    ]
    supports_amount = [len(heights) for heights in supports_height]

    # Yarding distances per corridor
    max_yarding_distance_per_cr = []
    average_yarding_distance_per_cr = []
    for line_idx, g in enumerate(grouped_class_indices[: len(rot_line_gdf)]):
        if len(g) == 0 or len(indices) == 0:
            max_yarding_distance_per_cr.append(0)
            average_yarding_distance_per_cr.append(0)
        else:
            dists = distance_carriage_support[g, line_idx]
            max_yarding_distance_per_cr.append(int(max(dists)))
            average_yarding_distance_per_cr.append(int(np.mean(dists)))

    # Tail spar (end mast) — read metadata but not coordinates (coords come from line end)
    endmast_height_list, endmast_BHD_list, endmast_max_holding_force_list = [], [], []
    endmast_x_list, endmast_y_list = [], []
    for _, row in rot_line_gdf.iterrows():
        end_tree = getattr(row, "end_support_tree", getattr(row, "end_anchor_tree", None))
        # coordinates: use geometry end point (not dict)
        end_pt = row.geometry.coords[-1]
        ex, ey = float(end_pt[0]), float(end_pt[1])

        # metadata from dict if present
        eh = eb = emf = 0
        if isinstance(end_tree, dict):
            eh = int(end_tree.get("h", 0))
            eb = int(end_tree.get("BHD", 0))
            emf = int(end_tree.get("max_holding_force", 0))

        endmast_height_list.append(eh)
        endmast_BHD_list.append(eb)
        endmast_max_holding_force_list.append(emf)
        endmast_x_list.append(round(ex, 2))
        endmast_y_list.append(round(ey, 2))

    # Road anchors (support dict or dataframe)
    road_anchor_height_list, road_anchor_BHD_list = [], []
    road_anchor_max_holding_force_list, road_anchor_x_list, road_anchor_y_list = [], [], []
    for _, row in rot_line_gdf.iterrows():
        ra = getattr(row, "road_anchor_tree_series", None)

        if isinstance(ra, dict):
            road_anchor_height_list.append(int(ra.get("h", 0)))
            road_anchor_BHD_list.append(int(ra.get("BHD", 0)))
            road_anchor_max_holding_force_list.append(int(ra.get("max_holding_force", 0)))
            road_anchor_x_list.append(round(float(ra.get("x", 0.0)), 2))
            road_anchor_y_list.append(round(float(ra.get("y", 0.0)), 2))
            continue

        if hasattr(ra, "iterrows"):
            try:
                first = next(ra.iterrows())[1]
                road_anchor_height_list.append(int(first.get("h", 0)))
                road_anchor_BHD_list.append(int(first.get("BHD", 0)))
                road_anchor_max_holding_force_list.append(int(first.get("max_holding_force", 0)))
                road_anchor_x_list.append(round(float(first.get("x", 0.0)), 2))
                road_anchor_y_list.append(round(float(first.get("y", 0.0)), 2))
            except StopIteration:
                road_anchor_height_list.append(0); road_anchor_BHD_list.append(0)
                road_anchor_max_holding_force_list.append(0); road_anchor_x_list.append(0.0); road_anchor_y_list.append(0.0)
            continue

        road_anchor_height_list.append(0); road_anchor_BHD_list.append(0)
        road_anchor_max_holding_force_list.append(0); road_anchor_x_list.append(0.0); road_anchor_y_list.append(0.0)

    # Global yarding distances
    max_yarding_distance = int(max(distance_trees_to_selected_lines)) if len(distance_trees_to_selected_lines) else 0
    average_yarding_distance = int(np.mean(distance_trees_to_selected_lines)) if len(distance_trees_to_selected_lines) else 0

    line_cost = int(sum(rot_line_gdf["line_cost"])) if len(rot_line_gdf) else 0
    total_cable_road_costs = int(line_cost + productivity_cost_overall)

    # Cost per m3
    denom = max(1, sum(wood_volume_per_cr) if len(wood_volume_per_cr) else 1)
    cost_per_m3 = round(total_cable_road_costs / denom, 2)

    # Ecological penalty
    if len(indices) > 0:
        ecological_penalty_threshold = 10
        ecological_penalty_lateral = np.where(
            distance_tree_line > ecological_penalty_threshold,
            distance_tree_line - ecological_penalty_threshold,
            0,
        )
        sum_eco_distances = int(
            sum(ecological_penalty_lateral[j][i] for i, j in zip(tree_to_line_assignment, range(len(ecological_penalty_lateral))))
        )
    else:
        sum_eco_distances = 0

    # Ergonomics penalty (double beyond threshold)
    if len(indices) > 0:
        ergonomics_penalty_threshold = 15
        ergonomic_penalty_lateral = np.where(
            distance_tree_line > ergonomics_penalty_threshold,
            (distance_tree_line - ergonomics_penalty_threshold) * 2,
            0,
        )
        sum_ergo_distances = int(
            sum(ergonomic_penalty_lateral[j][i] for i, j in zip(tree_to_line_assignment, range(len(ergonomic_penalty_lateral))))
        )
    else:
        sum_ergo_distances = 0

    # Volume per running meter
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


# -------------------------
# Internal helpers
# -------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sample_line_xy(line, min_points: int = 20, step: float = 5.0) -> Tuple[List[float], List[float]]:
    """Visual-only sampling along a corridor for smooth hover lines."""
    length = float(line.length)
    n_points = max(int(length // step) + 2, min_points)
    dists = np.linspace(0.0, length, n_points)
    xs, ys = [], []
    for d in dists:
        p = line.interpolate(d)
        xs.append(float(p.x)); ys.append(float(p.y))
    return xs, ys


def _labels_to_real_indices(sel_real: List[int], labels: List[int], fa_line_gdf: pd.DataFrame) -> List[Optional[int]]:
    """
    Convert per-tree labels in rot_line_gdf order -> REAL line indices.
    labels are 0..len(sel_real)-1; we map via the deterministic rot_line_gdf order.
    """
    if not sel_real or labels is None:
        return [None] * len(labels)
    rot_idx = list(fa_line_gdf[fa_line_gdf.index.isin(sel_real)].index)
    out: List[Optional[int]] = []
    for lbl in labels:
        try:
            il = int(lbl)
            out.append(int(rot_idx[il]) if 0 <= il < len(rot_idx) else None)
        except Exception:
            out.append(None)
    return out


# -------------------------
# Big class (precompute everything once)
# -------------------------
@dataclass
class VizData:
    forest_area_3: Any
    model_list: Any
    results_df: pd.DataFrame

    # core precomputed
    indices_to_show: List[int] = field(init=False)
    display_to_real: Dict[int, int] = field(init=False)
    real_to_display: Dict[int, int] = field(init=False)
    dtl_full: np.ndarray = field(init=False)  # tree↔line distances (full)
    dcs_full: np.ndarray = field(init=False)  # carriage↔support distances (full)
    palette: List[str] = field(init=False)

    # layouts precomputed up-front
    layout_union: Dict[str, Any] = field(init=False)
    layout_by_model: Dict[int, Dict[str, Any]] = field(init=False)

    # map payload (stable; uses length from line_gdf + volume from column or union layout)
    map: Dict[str, Any] = field(init=False)

    # overview table rows (all models)
    overview_rows: List[List[Any]] = field(init=False)

    def __post_init__(self):
        self._build_core()
        self._precompute_all_layouts()
        self._build_map_payload()
        self._build_overview_rows()

    # -------- core precompute --------
    def _build_core(self) -> None:
        self.indices_to_show = sorted({int(i) for row in self.results_df["selected_lines"] for i in row})
        display_names = list(range(1, len(self.indices_to_show) + 1))
        self.display_to_real = dict(zip(display_names, self.indices_to_show))
        self.real_to_display = dict(zip(self.indices_to_show, display_names))

        self.dtl_full, self.dcs_full = geometry_operations.compute_distances_facilities_clients(
            self.forest_area_3.harvesteable_trees_gdf,
            self.forest_area_3.line_gdf
        )
        self.palette = px.colors.qualitative.Plotly

    # -------- precompute update_layout_overview for union + every model --------
    def _precompute_all_layouts(self) -> None:
        self.layout_by_model = {}

        # 1) union layout
        if self.indices_to_show:
            self.layout_union = update_layout_overview(
                self.indices_to_show,
                self.forest_area_3,
                self.model_list,
                precomputed=(self.dtl_full, self.dcs_full),
            )
        else:
            self.layout_union = {}

        # 2) per-model layouts
        for i, res in self.results_df.iterrows():
            sel = list(map(int, res["selected_lines"]))
            self.layout_by_model[i] = update_layout_overview(
                sel,
                self.forest_area_3,
                self.model_list,
                precomputed=(self.dtl_full, self.dcs_full),
            )

    # -------- map data (stable; no per-selection recompute) --------
    def _compute_fixed_volumes_for_map(self) -> Dict[int, float]:
        """
        Pick wood volume per corridor for stable map hover.
        Priority:
          1) a column on line_gdf if present
          2) union layout (precomputed once)
          3) fallback to zeros
        """
        line = self.forest_area_3.line_gdf

        # 1) if a volume column exists on line_gdf, use it
        for col in ("wood_volume", "volume_m3", "volumen_m3", "volume"):
            if col in line.columns:
                return {int(i): _safe_float(line.loc[int(i), col]) for i in self.indices_to_show}

        # 2) union layout
        if self.indices_to_show and self.layout_union:
            rot_idx = list(line[line.index.isin(self.indices_to_show)].index)
            per_cr = self.layout_union.get("Wood Volume per Cable Corridor (m3)", [])
            return {int(r): _safe_float(v) for r, v in zip(rot_idx, per_cr)}

        # 3) fallback
        return {int(i): 0.0 for i in self.indices_to_show}

    def _build_map_payload(self) -> None:
        fa = self.forest_area_3
        idx = self.indices_to_show

        # Trees + BHD
        tree_x, tree_y, tree_bhd_cm = [], [], []
        gtrees = fa.harvesteable_trees_gdf
        bhd_series = gtrees.get("BHD", pd.Series([None] * len(gtrees)))
        for geom, bhd in zip(gtrees.geometry, bhd_series):
            tree_x.append(float(geom.x))
            tree_y.append(float(geom.y))
            tree_bhd_cm.append(None if pd.isna(bhd) else float(bhd))

        # Fixed volumes for map hover
        volumes_by_idx = self._compute_fixed_volumes_for_map()

        # ---- Tree assignment for the UNION (real indices per tree) ----
        union_labels = list(self.layout_union.get("Tree to Cable Corridor Assignment", [])) if self.layout_union else []
        tree_assignment_union = (
            _labels_to_real_indices(idx, union_labels, fa.line_gdf)
            if union_labels else [None] * len(gtrees)
        )

        # (Optional) per-model mapping ready if needed later
        # tree_assignment_by_model = {}
        # for i, res in self.results_df.iterrows():
        #     sel_real = list(map(int, res["selected_lines"]))
        #     labels = list(self.layout_by_model[i].get("Tree to Cable Corridor Assignment", []))
        #     tree_assignment_by_model[i] = _labels_to_real_indices(sel_real, labels, fa.line_gdf) if labels else [None] * len(gtrees)

        # Corridors (+ anchors)
        corridors: Dict[int, Dict[str, Any]] = {}
        subset = fa.line_gdf.loc[idx] if len(idx) else fa.line_gdf.iloc[[]]

        for real_idx, row in subset.iterrows():
            line = row.geometry
            xs, ys = _sample_line_xy(line)
            start_pt, end_pt = line.coords[0], line.coords[-1]

            length_m = _safe_float(row.get("line_length", 0.0))
            volume_m3 = _safe_float(volumes_by_idx.get(int(real_idx), 0.0))

            # Tail/end anchor:
            # Coordinates MUST come from the line end-point; only read BHD from dict if present.
            end_tree = getattr(row, "end_support_tree", getattr(row, "end_anchor_tree", None))
            ex, ey = float(end_pt[0]), float(end_pt[1])
            ebhd = 0.0
            if isinstance(end_tree, dict):
                ebhd = _safe_float(end_tree.get("BHD", 0))
            else:
                try:
                    ebhd = _safe_float(end_tree.get("BHD", 0))  # noqa: type: ignore
                except Exception:
                    pass

            # Road anchors
            road_anchor_entries: List[dict] = []
            ra_src = getattr(row, "road_anchor_tree_series", None)
            if isinstance(ra_src, pd.DataFrame) and not ra_src.empty:
                for _, r in ra_src.iterrows():
                    road_anchor_entries.append(dict(
                        x=_safe_float(r.get("x", 0)),
                        y=_safe_float(r.get("y", 0)),
                        BHD=_safe_float(r.get("BHD", 0)),
                    ))
            elif isinstance(ra_src, dict):
                road_anchor_entries.append(dict(
                    x=_safe_float(ra_src.get("x", 0)),
                    y=_safe_float(ra_src.get("y", 0)),
                    BHD=_safe_float(ra_src.get("BHD", 0)),
                ))

            corridors[int(real_idx)] = dict(
                xs=xs, ys=ys,
                start=(float(start_pt[0]), float(start_pt[1])),
                end=(float(end_pt[0]), float(end_pt[1])),
                tail_anchor=dict(x=ex, y=ey, BHD=ebhd),
                road_anchors=road_anchor_entries,
                length_m=length_m,
                volume_m3=volume_m3,
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

        color_map = {rid: self.palette[i % len(self.palette)] for i, rid in enumerate(self.indices_to_show)}

        self.map = dict(
            tree_x=tree_x,
            tree_y=tree_y,
            tree_bhd_cm=tree_bhd_cm,
            tree_assignment=tree_assignment_union,  # <-- needed by map.py to color trees
            corridors=corridors,
            indices_to_show=self.indices_to_show,
            display_to_real=self.display_to_real,
            real_to_display=self.real_to_display,
            color_map=color_map,
            x_range=x_range,
            y_range=y_range,
            dtl_full=self.dtl_full,
            full_line_index=np.asarray(self.forest_area_3.line_gdf.index, dtype=int),
            map_title="",
        )

    # -------- overview (all models) --------
    def _build_overview_rows(self) -> None:
        """Build the overview table rows from precomputed per-model layouts (no recomputation)."""
        rows: List[List[Any]] = []
        for i, res in self.results_df.iterrows():
            sel = list(map(int, res["selected_lines"]))
            layout = self.layout_by_model[i]

            rows.append([
                i + 1,  # "Modell"
                layout.get("Total Cable Corridor Costs (€)"),
                layout.get("Setup and Takedown, Prod. Costs (€)"),
                layout.get("Ecol. Penalty"),
                layout.get("Ergon. Penalty"),
                [self.real_to_display.get(int(idx), int(idx)) for idx in sel],
                layout.get("Max lateral Yarding Distance (m)"),
                layout.get("Average lateral Yarding Distance (m)"),
                int(np.mean(layout["Supports Amount"])) if layout.get("Supports Amount") else 0,
                layout.get("Cost per m3 (€)"),
                layout.get("Volume per Meter (m3/m)"),
            ])
        self.overview_rows = rows

    # -------- per-selection “detail” tables (no recompute) --------
    def selected_rows(self, selected_index: int) -> List[List[str]]:
        """
        Returns rows that match:
        ["Seiltrassen Nummer","Aufbaukosten [€]","Seillänge [m]","Vfm pro Seiltrasse [m³]",
         "Stützbaum Anzahl","Tragseilhöhe Stütze [m]","Durchschnittliche Baumhöhe [m]",
         "Max Zugseillänge [m]","Durchschnittliche Zugseillänge [m]"]
        """
        if selected_index < 0 or selected_index >= len(self.results_df):
            return []

        sel_real = list(map(int, self.results_df.iloc[selected_index]["selected_lines"]))
        layout = self.layout_by_model[selected_index]

        vols        = layout.get("Wood Volume per Cable Corridor (m3)", [])
        sup_count   = layout.get("Supports Amount", [])
        sup_heights = layout.get("Supports Height (m)", [])
        avg_tree_h  = layout.get("Average Tree Height (m)", [])
        max_yard    = layout.get("Max Yarding Distance per Cable Corridor (m)", [])
        avg_yard    = layout.get("Average Yarding Distance per Cable Corridor (m)", [])

        fa = self.forest_area_3
        subset = fa.line_gdf.loc[fa.line_gdf.index.isin(sel_real)].loc[sel_real]

        rows: List[List[str]] = []
        for i, real_idx in enumerate(sel_real):
            disp_id = self.real_to_display.get(int(real_idx), int(real_idx))

            line_cost   = int(subset.loc[real_idx, "line_cost"])   if "line_cost"   in subset.columns else 0
            line_length = int(subset.loc[real_idx, "line_length"]) if "line_length" in subset.columns else 0

            vol    = int(vols[i])               if i < len(vols)        else 0
            s_cnt  = int(sup_count[i])          if i < len(sup_count)   else 0
            s_hlst = sup_heights[i]             if i < len(sup_heights) and isinstance(sup_heights[i], list) else []
            s_hstr = "/" if not s_hlst else ", ".join(str(int(h)) for h in s_hlst)

            avg_h  = float(avg_tree_h[i])       if i < len(avg_tree_h)  else 0.0
            max_y  = int(max_yard[i])           if i < len(max_yard)    else 0
            avg_y  = int(avg_yard[i])           if i < len(avg_yard)    else 0

            rows.append([
                str(disp_id),
                str(line_cost),
                str(line_length),
                str(vol),
                str(s_cnt),
                s_hstr,
                f"{avg_h:.2f}",
                str(max_y),
                str(avg_y),
            ])

        return rows

    def anchor_rows(self, selected_index: int) -> List[List[str]]:
        """
        Returns rows that match:
        ["Seiltrassen Nummer","BHD [cm]","Height [m]","X-Koordinate","Y-Koordinate"]
        Uses the tail/end anchor (Endmast).
        """
        if selected_index < 0 or selected_index >= len(self.results_df):
            return []

        sel_real = list(map(int, self.results_df.iloc[selected_index]["selected_lines"]))
        fa = self.forest_area_3
        subset = fa.line_gdf.loc[fa.line_gdf.index.isin(sel_real)].loc[sel_real]

        rows: List[List[str]] = []
        for real_idx, row in subset.iterrows():
            disp_id = self.real_to_display.get(int(real_idx), int(real_idx))

            ta = getattr(row, "end_support_tree", getattr(row, "end_anchor_tree", None))
            bhd = h = x = y = None
            if isinstance(ta, dict):
                bhd = int(ta.get("BHD", 0))
                h   = int(ta.get("h", 0))
                x   = round(float(ta.get("x", 0.0)), 2)
                y   = round(float(ta.get("y", 0.0)), 2)

            rows.append([
                str(disp_id),
                "" if bhd is None else str(bhd),
                "" if h   is None else str(h),
                "" if x   is None else str(x),
                "" if y   is None else str(y),
            ])

        return rows

    # -------- radar scores --------
    def make_radar_scores(self, axes: List[str])-> pd.DataFrame:
        df = self.results_df.copy()

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
        scores["color"] = [_convert_hex_to_rgba(c) for c in colors]
        scores["fill_color"] = [_convert_hex_to_rgba(c, 0.18) for c in colors]
        scores["raw_eco"] = self.results_df.loc[df.index, "ecological_distances_RNI"]
        scores["raw_ergo"] = self.results_df.loc[df.index, "ergonomics_distances_RNI"]
        scores["raw_cost"] = self.results_df.loc[df.index, "cost_objective_RNI"]

        scores["triangle_area"] = scores.apply(_triangle_area_on_axes, axis=1, args=(axes,))
        return scores

    def to_string(self, full: bool=False, max_list: int=30, max_dict_items: int=30, max_df_rows: int=20, max_df_cols: int=20) -> str:
        from textwrap import indent

        def fmt_ndarray(a: np.ndarray) -> str:
            if full:
                return f"ndarray(shape={a.shape}, dtype={a.dtype})\n" + indent(str(a), "  ")
            return f"ndarray(shape={a.shape}, dtype={a.dtype}, sample={np.array2string(a.flatten()[:min(20, a.size)])})"

        def fmt_df(df: pd.DataFrame) -> str:
            if full:
                return f"DataFrame(shape={df.shape}):\n" + indent(df.to_string(), "  ")
            cols = list(df.columns)[:max_df_cols]
            head = df[cols].head(max_df_rows)
            return f"DataFrame(shape={df.shape}, cols={list(df.columns)})\n" + indent(head.to_string(index=False), "  ")

        def fmt_series(s: pd.Series) -> str:
            if full:
                return f"Series(len={len(s)}, name={s.name}):\n" + indent(s.to_string(), "  ")
            return f"Series(len={len(s)}, name={s.name}, head={s.head(min(max_df_rows, len(s))).to_list()})"

        def fmt_list(x: list) -> str:
            if full or len(x) <= max_list:
                return str(x)
            return f"{x[:max_list]} … (len={len(x)})"

        def fmt_dict(d: dict) -> str:
            if full or len(d) <= max_dict_items:
                parts = []
                for k, v in d.items():
                    parts.append(f"{k}: {fmt_any(v)}")
                return "{\n" + indent("\n".join(parts), "  ") + "\n}"
            some = list(d.items())[:max_dict_items]
            parts = [f"{k}: {fmt_any(v)}" for k, v in some]
            return "{\n" + indent("\n".join(parts), "  ") + f"\n  … ({len(d)} items total)\n)" + "}"

        def fmt_any(v) -> str:
            try:
                if isinstance(v, np.ndarray):
                    return fmt_ndarray(v)
                if isinstance(v, pd.DataFrame):
                    return fmt_df(v)
                if isinstance(v, pd.Series):
                    return fmt_series(v)
                if isinstance(v, dict):
                    return fmt_dict(v)
                if isinstance(v, list):
                    return fmt_list(v)
                if isinstance(v, tuple):
                    return "(" + ", ".join(fmt_any(t) for t in v) + ")"
                return repr(v)
            except Exception as e:
                return f"<unprintable: {e}>"

        lines = []
        lines.append(f"{self.__class__.__name__}" + "{{")

        # Core
        lines.append("  # Core")
        lines.append(f"  indices_to_show: {fmt_list(self.indices_to_show)}")
        lines.append(f"  display_to_real: {fmt_dict(self.display_to_real)}")
        lines.append(f"  real_to_display: {fmt_dict(self.real_to_display)}")
        lines.append(f"  dtl_full: {fmt_any(self.dtl_full)}")
        lines.append(f"  dcs_full: {fmt_any(self.dcs_full)}")
        lines.append(f"  palette: {fmt_list(self.palette)}")

        # Layouts
        lines.append("\n  # Layouts")
        lines.append("  layout_union: " + indent(fmt_dict(self.layout_union), "    "))
        lines.append(f"  layout_by_model: dict(len={len(self.layout_by_model)})")
        if self.layout_by_model:
            first_key = next(iter(self.layout_by_model.keys()))
            lines.append(f"    sample[{first_key}]: " + indent(fmt_dict(self.layout_by_model[first_key]), "      "))

        # Map payload
        lines.append("\n  # Map")
        lines.append("  map: {")
        for k in ("tree_x", "tree_y", "tree_bhd_cm", "tree_assignment", "corridors", "indices_to_show",
                  "display_to_real", "real_to_display", "color_map", "x_range", "y_range", "map_title"):
            v = self.map.get(k, None)
            if k in ("tree_x", "tree_y", "tree_bhd_cm", "tree_assignment") and isinstance(v, list):
                v_str = fmt_list(v)
            elif k in ("display_to_real", "real_to_display", "color_map") and isinstance(v, dict):
                v_str = fmt_dict(v)
            else:
                v_str = fmt_any(v)
            lines.append(f"    {k}: {v_str}")
        lines.append("  }")

        # Overview rows
        lines.append("\n  # Overview rows")
        if full:
            lines.append("  " + fmt_list(self.overview_rows))
        else:
            n = len(self.overview_rows)
            sample = self.overview_rows[:min(5, n)]
            lines.append(f"  rows={n}, sample={sample}")

        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_string(full=False)


# -------------------------
# Public factory
# -------------------------
def build_viz_data(forest_area_3, model_list, results_df: pd.DataFrame) -> VizData:
    """
    Single entry point for the UI. Computes:
      • distances (once)
      • layout_overview for union and for every model (once)
      • stable map payload (length + fixed volume + tree assignments)
      • overview rows (all models)
    """
    return VizData(forest_area_3=forest_area_3, model_list=model_list, results_df=results_df)


# ----- Legacy wrappers (keep if interface imports these names) -----
def prepare_map_data(forest_area_3, results_df: pd.DataFrame, model_list=None) -> Dict[str, Any]:
    return build_viz_data(forest_area_3, model_list, results_df).map

def get_overview_table_data(forest_area_3, model_list, results_df: pd.DataFrame) -> List[List[Any]]:
    return build_viz_data(forest_area_3, model_list, results_df).overview_rows

def get_selected_table_data(forest_area_3, model_list, results_df: pd.DataFrame, selected_index: int) -> List[List[Any]]:
    return build_viz_data(forest_area_3, model_list, results_df).selected_rows(selected_index)

def get_anchor_table_data(forest_area_3, model_list, results_df: pd.DataFrame, selected_index: int) -> List[List[str]]:
    return build_viz_data(forest_area_3, model_list, results_df).anchor_rows(selected_index)

def make_radar_scores(results_df: pd.DataFrame, axes: List[str]) -> pd.DataFrame:
    dummy = VizData(forest_area_3=None, model_list=None, results_df=results_df)  # only uses results_df
    return dummy.make_radar_scores(axes)
