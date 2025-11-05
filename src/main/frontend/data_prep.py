from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.colors import hex_to_rgb

from src.main import geometry_operations


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def _scale(series: pd.Series, pad_low: float = 0.1) -> pd.Series:
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


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sample_line_xy(line, min_points: int = 20, step: float = 5.0) -> Tuple[List[float], List[float]]:
    """
    Take a shapely LineString-like geometry and sample along it for plotting.
    This smooths our display of cable corridors.
    """
    length = float(line.length)
    n_points = max(int(length // step) + 2, min_points)
    dists = np.linspace(0.0, length, n_points)
    xs, ys = [], []
    for d in dists:
        p = line.interpolate(d)
        xs.append(float(p.x))
        ys.append(float(p.y))
    return xs, ys


# --------------------------------------------------------------------------------------
# NEW: tree-color helpers (mirrors interface.py behavior)
# --------------------------------------------------------------------------------------

_PLOTLY = px.colors.qualitative.Plotly


def _labels_to_plotly_colors(labels: List[int]) -> List[str]:
    """
    Exactly how interface.py colors trees:
      for i in labels -> px.colors.qualitative.Plotly[i % len(palette)]
    """
    out = []
    n = len(_PLOTLY)
    for lab in labels:
        try:
            out.append(_PLOTLY[int(lab) % n])
        except Exception:
            out.append("green")
    return out


def _tree_colors_for_indices(
    indices: List[int],
    forest_area_3,
    dtl_full: np.ndarray,
) -> List[str]:
    """
    Compute per-tree colors for a *specific* set of REAL cable corridor indices.
    - We slice dtl_full (trees x ALL corridors) down to (trees x selected corridors),
      preserving the order of `indices`.
    - Then argmin over that slice gives a label in [0 .. len(indices)-1].
    - Then map label -> Plotly color index.
    """
    num_trees = len(forest_area_3.harvesteable_trees_gdf)

    if not indices:
        # No corridors selected: everything pure green.
        return ["green"] * num_trees

    # Build a lookup real_idx -> column position in the full line_gdf order
    full_idx = forest_area_3.line_gdf.index
    pos_map: Dict[int, int] = {int(k): i for i, k in enumerate(full_idx)}

    # Keep only valid indices that exist in line_gdf
    valid_real_indices: List[int] = []
    valid_cols: List[int] = []
    for ridx in indices:
        ridx_int = int(ridx)
        if ridx_int in pos_map:
            valid_real_indices.append(ridx_int)
            valid_cols.append(pos_map[ridx_int])

    if not valid_cols:
        return ["green"] * num_trees

    # Slice dtl_full to just those corridors
    dtl_slice = dtl_full[:, valid_cols]

    # argmin -> label 0..k-1, where k=len(valid_real_indices)
    try:
        labels = np.argmin(dtl_slice, axis=1).astype(int).tolist()
    except Exception:
        # fallback green if something impossible happens
        return ["green"] * num_trees

    # Map 0..k-1 -> palette
    return _labels_to_plotly_colors(labels)


# --------------------------------------------------------------------------------------
# Tree->corridor assignment real-index helper
# --------------------------------------------------------------------------------------

def _labels_to_real_indices(
    sel_real: List[int],
    labels: List[int],
) -> List[Optional[int]]:
    """
    Convert per-tree labels (0..len(sel_real)-1) to the REAL corridor index.
    If label is 0, that's sel_real[0], etc.
    Returns a list, one entry per tree, each element = REAL corridor index or None.
    """
    if not sel_real or labels is None:
        return [None] * len(labels)

    out: List[Optional[int]] = []
    n = len(sel_real)
    for lbl in labels:
        try:
            il = int(lbl)
            if 0 <= il < n:
                out.append(int(sel_real[il]))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


# --------------------------------------------------------------------------------------
# update_layout_overview (robust to bad indices)
# --------------------------------------------------------------------------------------

def update_layout_overview(indices, forest_area_3, model_list, precomputed=None) -> dict:
    """
    Compute metrics for the given selection of cable corridors (REAL indices).
    Uses optional precomputed distance matrices for speed.
    Robust to invalid / missing indices.
    """
    line_gdf = forest_area_3.line_gdf

    # position map: real_idx -> column index in full dtl_full/dcs_full
    full_index = line_gdf.index
    pos_map: Dict[int, int] = {int(k): i for i, k in enumerate(full_index)}

    # Filter to only indices that actually exist in line_gdf
    sel_real: List[int] = []
    for i in indices:
        ii = int(i)
        if ii in pos_map:
            sel_real.append(ii)

    # If nothing valid, return an empty-ish layout
    if not sel_real:
        return {
            "Wood Volume per Cable Corridor (m3)": [],
            "Total Cable Corridor Costs (€)": 0,
            "Setup and Takedown, Prod. Costs (€)": "0 / 0",
            "Ecol. Penalty": 0,
            "Ergon. Penalty": 0,
            "Tree to Cable Corridor Assignment": [0] * len(forest_area_3.harvesteable_trees_gdf),
            "Supports Height (m)": [],
            "Supports Amount": [],
            "Max lateral Yarding Distance (m)": 0,
            "Average lateral Yarding Distance (m)": 0,
            "Cost per m3 (€)": 0.0,
            "Average Tree Height (m)": [],
            "Volume per Meter (m3/m)": 0.0,
            "Max Yarding Distance per Cable Corridor (m)": [],
            "Average Yarding Distance per Cable Corridor (m)": [],
            "Anchor height": [],
            "Anchor BHD": [],
            "Anchor max holding force": [],
            "Anchor x coordinate": [],
            "Anchor y coordinate": [],
            "Corresponding Cable Corridor": [],
            "Road Anchor height": [],
            "Road Anchor BHD": [],
            "Road Anchor max holding force": [],
            "Road Anchor x coordinate": [],
            "Road Anchor y coordinate": [],
            "Road Anchor Angle of Attack": [],
            "Tail Anchor Angle of Attack": [],
        }

    # Slice line_gdf down to sel_real (in that exact order)
    rot_line_gdf = line_gdf.loc[sel_real]

    # Distances slice
    if precomputed is not None:
        dtl_full, dcs_full = precomputed
        cols = [pos_map[i] for i in sel_real]
        distance_tree_line = dtl_full[:, cols]  # trees x selected_lines
        distance_carriage_support = dcs_full[:, cols]
    else:
        distance_tree_line, distance_carriage_support = geometry_operations.compute_distances_facilities_clients(
            forest_area_3.harvesteable_trees_gdf,
            rot_line_gdf
        )

    # Assign each tree to closest selected cable corridor
    try:
        tree_to_line_assignment = np.argmin(distance_tree_line, axis=1)
        distance_trees_to_selected_lines = distance_tree_line[
            range(distance_tree_line.shape[0]), tree_to_line_assignment
        ]
    except Exception:
        tree_to_line_assignment = np.zeros((len(forest_area_3.harvesteable_trees_gdf),), dtype=int)
        distance_trees_to_selected_lines = np.zeros_like(tree_to_line_assignment, dtype=float)

    # Productivity costs
    if model_list is not None and hasattr(model_list[0], "productivity_cost"):
        prod = model_list[0].productivity_cost
        sel_cols = [pos_map[i] for i in sel_real]
        selected_prod_cost = prod[:, sel_cols]
    else:
        selected_prod_cost = np.zeros((len(forest_area_3.harvesteable_trees_gdf), len(sel_real)))

    productivity_cost_overall = 0
    for index, val in enumerate(tree_to_line_assignment):
        col = min(int(val), selected_prod_cost.shape[1] - 1)
        productivity_cost_overall += selected_prod_cost[index][col]

    # Group trees per label
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(max(1, len(sel_real)))
    ]

    # Wood volume per corridor
    gtrees = forest_area_3.harvesteable_trees_gdf
    wood_volume_per_cr = [
        int(sum(gtrees.iloc[g]["cubic_volume"])) if len(g) else 0
        for g in grouped_class_indices
    ][: len(sel_real)]

    # Avg tree height per corridor
    average_tree_size_per_cr = [
        round(float(sum(gtrees.iloc[g]["h"])) / len(g), 2) if len(g) else 0.0
        for g in grouped_class_indices
    ][: len(sel_real)]

    # Supports
    supports_height = [
        (
            [segment.start_support.attachment_height for segment in cr_object.supported_segments[1:]]
            if cr_object.supported_segments else []
        )
        for cr_object in rot_line_gdf["Cable Road Object"]
    ]
    supports_amount = [len(heights) for heights in supports_height]

    # Per-corridor yarding distances
    max_yarding_distance_per_cr, average_yarding_distance_per_cr = [], []
    for line_idx, g in enumerate(grouped_class_indices[: len(sel_real)]):
        if len(g) == 0:
            max_yarding_distance_per_cr.append(0)
            average_yarding_distance_per_cr.append(0)
        else:
            dists = distance_carriage_support[g, line_idx]
            max_yarding_distance_per_cr.append(int(max(dists)))
            average_yarding_distance_per_cr.append(int(np.mean(dists)))

    # Tail anchor/end support info (from END point of line)
    endmast_height_list, endmast_BHD_list, endmast_max_holding_force_list = [], [], []
    endmast_x_list, endmast_y_list = [], []
    for _, row in rot_line_gdf.iterrows():
        end_tree = getattr(row, "end_support_tree", getattr(row, "end_anchor_tree", None))
        end_pt = row.geometry.coords[-1]
        ex, ey = float(end_pt[0]), float(end_pt[1])

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

    # Road anchors (first anchor per line OR fallback zeros for overview table)
    road_anchor_height_list, road_anchor_BHD_list = [], []
    road_anchor_max_holding_force_list, road_anchor_x_list, road_anchor_y_list = [], [], []
    for _, row in rot_line_gdf.iterrows():
        ra = getattr(row, "road_anchor_tree_series", None)

        if isinstance(ra, dict):
            road_anchor_height_list.append(int(ra.get("h", 0)))
            road_anchor_BHD_list.append(int(ra.get("BHD", 0)))
            road_anchor_max_holding_force_list.append(int(ra.get("max_holding_force", 0)))
            road_anchor_x_list.append(round(_safe_float(ra.get("x", 0.0)), 2))
            road_anchor_y_list.append(round(_safe_float(ra.get("y", 0.0)), 2))
            continue

        if hasattr(ra, "iterrows"):
            try:
                first = next(ra.iterrows())[1]
                road_anchor_height_list.append(int(first.get("h", 0)))
                road_anchor_BHD_list.append(int(first.get("BHD", 0)))
                road_anchor_max_holding_force_list.append(int(first.get("max_holding_force", 0)))
                road_anchor_x_list.append(round(_safe_float(first.get("x", 0.0)), 2))
                road_anchor_y_list.append(round(_safe_float(first.get("y", 0.0)), 2))
            except StopIteration:
                road_anchor_height_list.append(0)
                road_anchor_BHD_list.append(0)
                road_anchor_max_holding_force_list.append(0)
                road_anchor_x_list.append(0.0)
                road_anchor_y_list.append(0.0)
            continue

        road_anchor_height_list.append(0)
        road_anchor_BHD_list.append(0)
        road_anchor_max_holding_force_list.append(0)
        road_anchor_x_list.append(0.0)
        road_anchor_y_list.append(0.0)

    # Overall yarding distance stats
    if len(distance_trees_to_selected_lines) > 0:
        max_yarding_distance = int(max(distance_trees_to_selected_lines))
        average_yarding_distance = int(np.mean(distance_trees_to_selected_lines))
    else:
        max_yarding_distance = 0
        average_yarding_distance = 0

    # cost
    line_cost_total = int(sum(rot_line_gdf["line_cost"])) if len(rot_line_gdf) else 0
    total_cable_road_costs = int(line_cost_total + productivity_cost_overall)

    denom = max(1, sum(wood_volume_per_cr) if len(wood_volume_per_cr) else 1)
    cost_per_m3 = round(total_cable_road_costs / denom, 2)

    # ecological penalty
    if len(sel_real) > 0:
        threshold_eco = 10
        eco_penalty_lateral = np.where(
            distance_tree_line > threshold_eco,
            distance_tree_line - threshold_eco,
            0,
        )
        sum_eco_distances = int(
            sum(eco_penalty_lateral[j][i] for i, j in zip(tree_to_line_assignment, range(len(eco_penalty_lateral))))
        )
    else:
        sum_eco_distances = 0

    # ergonomics penalty
    if len(sel_real) > 0:
        threshold_ergo = 15
        ergo_penalty_lateral = np.where(
            distance_tree_line > threshold_ergo,
            (distance_tree_line - threshold_ergo) * 2,
            0,
        )
        sum_ergo_distances = int(
            sum(ergo_penalty_lateral[j][i] for i, j in zip(tree_to_line_assignment, range(len(ergo_penalty_lateral))))
        )
    else:
        sum_ergo_distances = 0

    # volume per running meter
    total_len = float(sum(rot_line_gdf["line_length"])) if len(rot_line_gdf) else 0.0
    volume_per_meter = round((sum(wood_volume_per_cr) / total_len) if total_len else 0.0, 2)

    return {
        "Wood Volume per Cable Corridor (m3)": wood_volume_per_cr,
        "Total Cable Corridor Costs (€)": total_cable_road_costs,
        "Setup and Takedown, Prod. Costs (€)": f"{line_cost_total} / {int(productivity_cost_overall)}",
        "Ecol. Penalty": sum_eco_distances,
        "Ergon. Penalty": sum_ergo_distances,
        "Tree to Cable Corridor Assignment": tree_to_line_assignment
        if len(sel_real) > 0
        else [0] * len(forest_area_3.harvesteable_trees_gdf),
        "Supports Height (m)": supports_height,
        "Supports Amount": supports_amount,
        "Max lateral Yarding Distance (m)": max_yarding_distance,
        "Average lateral Yarding Distance (m)": average_yarding_distance,
        "Cost per m3 (€)": cost_per_m3,
        "Average Tree Height (m)": average_tree_size_per_cr,
        "Volume per Meter (m3/m)": volume_per_meter,
        "Max Yarding Distance per Cable Corridor (m)": max_yarding_distance_per_cr,
        "Average Yarding Distance per Cable Corridor (m)": average_yarding_distance_per_cr,
        "Anchor height": endmast_height_list,
        "Anchor BHD": endmast_BHD_list,
        "Anchor max holding force": endmast_max_holding_force_list,
        "Anchor x coordinate": endmast_x_list,
        "Anchor y coordinate": endmast_y_list,
        "Corresponding Cable Corridor": sel_real,
        "Road Anchor height": road_anchor_height_list,
        "Road Anchor BHD": road_anchor_BHD_list,
        "Road Anchor max holding force": road_anchor_max_holding_force_list,
        "Road Anchor x coordinate": road_anchor_x_list,
        "Road Anchor y coordinate": road_anchor_y_list,
        "Road Anchor Angle of Attack": rot_line_gdf["angle_between_start_support_and_cr"] if len(rot_line_gdf) else [],
        "Tail Anchor Angle of Attack": rot_line_gdf["angle_between_end_support_and_cr"] if len(rot_line_gdf) else [],
    }


# --------------------------------------------------------------------------------------
# VizData
# --------------------------------------------------------------------------------------

@dataclass
class VizData:
    forest_area_3: Any
    model_list: Any
    results_df: pd.DataFrame

    # core
    indices_to_show: List[int] = field(init=False)
    display_to_real: Dict[int, int] = field(init=False)
    real_to_display: Dict[int, int] = field(init=False)
    dtl_full: np.ndarray = field(init=False)
    dcs_full: np.ndarray = field(init=False)
    palette: List[str] = field(init=False)

    # layouts
    layout_union: Dict[str, Any] = field(init=False)
    layout_by_model: Dict[int, Dict[str, Any]] = field(init=False)

    # map payload
    map: Dict[str, Any] = field(init=False)

    # overview table rows
    overview_rows: List[List[Any]] = field(init=False)

    def __post_init__(self):
        self._build_core()
        self._precompute_all_layouts()
        self._build_map_payload()
        self._build_overview_rows()

    def _build_core(self) -> None:
        """
        - derive indices_to_show (unique real corridor indices across all models)
        - compute full distance matrices dtl_full/dcs_full
        - set up palettes and mappings (display<->real)
        with robust filtering against forest_area_3.line_gdf existing indices
        """
        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))

        flat_ids: List[int] = []
        for row in self.results_df["selected_lines"]:
            for ridx in row:
                ii = int(ridx)
                if ii in valid_line_ids:
                    flat_ids.append(ii)

        self.indices_to_show = sorted(set(flat_ids))

        display_names = list(range(1, len(self.indices_to_show) + 1))
        self.display_to_real = dict(zip(display_names, self.indices_to_show))
        self.real_to_display = dict(zip(self.indices_to_show, display_names))

        # Precompute distances tree->all lines, carriage->all lines
        self.dtl_full, self.dcs_full = geometry_operations.compute_distances_facilities_clients(
            self.forest_area_3.harvesteable_trees_gdf,
            self.forest_area_3.line_gdf,
        )

        # Palette used everywhere
        self.palette = list(px.colors.qualitative.Plotly)

    def _precompute_all_layouts(self) -> None:
        """
        Build:
        - layout_union for global indices_to_show
        - layout_by_model[i] for each row in results_df
        """
        self.layout_by_model = {}

        if self.indices_to_show:
            self.layout_union = update_layout_overview(
                self.indices_to_show,
                self.forest_area_3,
                self.model_list,
                precomputed=(self.dtl_full, self.dcs_full),
            )
        else:
            self.layout_union = {}

        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))

        for i, res in self.results_df.iterrows():
            sel_real = [int(x) for x in res["selected_lines"] if int(x) in valid_line_ids]
            self.layout_by_model[i] = update_layout_overview(
                sel_real,
                self.forest_area_3,
                self.model_list,
                precomputed=(self.dtl_full, self.dcs_full),
            )

    def _compute_fixed_volumes_for_map(self) -> Dict[int, float]:
        """
        Map real corridor index -> volume for hover.
        First try line_gdf columns like 'wood_volume' / 'volume_m3' etc.
        Fallback: use layout_union's "Wood Volume per Cable Corridor (m3)".
        """
        line = self.forest_area_3.line_gdf

        # try direct volume columns in line_gdf
        for col in ("wood_volume", "volume_m3", "volumen_m3", "volume"):
            if col in line.columns:
                return {
                    int(i): _safe_float(line.loc[int(i), col])
                    for i in self.indices_to_show
                }

        # fallback to layout_union values
        if self.indices_to_show and self.layout_union:
            sel_real = list(self.layout_union.get("Corresponding Cable Corridor", self.indices_to_show))
            per_cr = self.layout_union.get("Wood Volume per Cable Corridor (m3)", [])
            return {
                int(r): _safe_float(v)
                for r, v in zip(sel_real, per_cr)
            }

        # final fallback
        return {int(i): 0.0 for i in self.indices_to_show}

    def _build_map_payload(self) -> None:
        """
        Build the full map payload that Map(...) uses.
        Includes:
          - trees coords, BHD
          - all corridor polylines, anchors, lengths, volumes
          - consistent bbox (from interface.py logic)
          - precomputed tree colors for:
                * default (all green)
                * union selection
                * each model row
                * arbitrary selection (tuple of real indices, ordered)
        """
        fa = self.forest_area_3
        idx_all = self.indices_to_show

        # --- Trees ---
        gtrees = fa.harvesteable_trees_gdf
        tree_x = [float(geom.x) for geom in gtrees.geometry]
        tree_y = [float(geom.y) for geom in gtrees.geometry]

        bhd_series = gtrees.get("BHD", pd.Series([None] * len(gtrees)))
        tree_bhd_cm = [None if pd.isna(b) else float(b) for b in bhd_series]

        # --- Corridors/anchors ---
        volumes_by_idx = self._compute_fixed_volumes_for_map()
        corridors: Dict[int, Dict[str, Any]] = {}

        subset = fa.line_gdf.loc[idx_all] if len(idx_all) else fa.line_gdf.iloc[[]]

        display_lookup = {
            int(real): int(self.real_to_display.get(int(real), int(real)))
            for real in self.indices_to_show
        }

        for real_idx, row in subset.iterrows():
            line = row.geometry
            xs, ys = _sample_line_xy(line)
            start_pt, end_pt = line.coords[0], line.coords[-1]

            length_m = _safe_float(row.get("line_length", 0.0))
            volume_m3 = _safe_float(volumes_by_idx.get(int(real_idx), 0.0))

            display_id = display_lookup.get(int(real_idx), int(real_idx))

            # Tail anchor from END coords, BHD from end_* dict if available
            end_tree = getattr(row, "end_anchor_tree", None)
            ex, ey = float(end_tree.loc["x"]), float(end_tree.loc["y"])
            ebhd = end_tree.loc["BHD"]

            # Road anchors: gather ALL anchors available
            road_anchors_list: List[dict] = []
            ra_src = getattr(row, "road_anchor_tree_series", None)

            if isinstance(ra_src, pd.DataFrame) and not ra_src.empty:
                for _, r in ra_src.iterrows():
                    road_anchors_list.append(
                        dict(
                            x=_safe_float(r.get("x", 0)),
                            y=_safe_float(r.get("y", 0)),
                            BHD=_safe_float(r.get("BHD", 0)),
                        )
                    )
            elif isinstance(ra_src, dict) and "features" in ra_src:
                # geojson-ish style
                for feat in ra_src["features"]:
                    props = feat.get("properties", {})
                    road_anchors_list.append(
                        dict(
                            x=_safe_float(props.get("x", 0)),
                            y=_safe_float(props.get("y", 0)),
                            BHD=_safe_float(props.get("BHD", 0)),
                        )
                    )
            elif isinstance(ra_src, dict):
                road_anchors_list.append(
                    dict(
                        x=_safe_float(ra_src.get("x", 0)),
                        y=_safe_float(ra_src.get("y", 0)),
                        BHD=_safe_float(ra_src.get("BHD", 0)),
                    )
                )

            corridors[int(real_idx)] = dict(
                xs=xs,
                ys=ys,
                start=(float(start_pt[0]), float(start_pt[1])),
                end=(float(end_pt[0]), float(end_pt[1])),
                tail_anchor=dict(x=ex, y=ey, BHD=ebhd),
                road_anchors=road_anchors_list,
                length_m=length_m,
                volume_m3=volume_m3,
                display_id=display_id,
            )

        # ----------------------------------------------------------------------------------
        # Compute map extents like interface.py:
        #   - take all corridor line geometries
        #   - include tail anchors & road anchors
        #   - small fixed padding (10m)
        #   => gives a nice tight frame instead of huge blank space
        # ----------------------------------------------------------------------------------
        x_vals = []
        y_vals = []

        # 1. all cable road line vertices
        for geom in fa.line_gdf.geometry:
            try:
                xs_geom, ys_geom = geom.xy
                x_vals.extend(xs_geom)
                y_vals.extend(ys_geom)
            except Exception:
                pass

        # 2. all tail / end support trees
        # try both end_support_tree and end_anchor_tree in case dataset uses one or the other
        if hasattr(fa.line_gdf, "end_support_tree"):
            for tail in fa.line_gdf.end_support_tree:
                if isinstance(tail, pd.DataFrame) and not tail.empty:
                    x_vals.extend(list(tail["x"].astype(float)))
                    y_vals.extend(list(tail["y"].astype(float)))
                elif isinstance(tail, dict) and "features" in tail:
                    for f in tail["features"]:
                        props = f.get("properties", f)
                        x_vals.append(float(props.get("x", 0)))
                        y_vals.append(float(props.get("y", 0)))
                elif isinstance(tail, dict):
                    # single dict with x,y
                    if "x" in tail and "y" in tail:
                        x_vals.append(float(tail["x"]))
                        y_vals.append(float(tail["y"]))
        if hasattr(fa.line_gdf, "end_anchor_tree"):
            for tail in fa.line_gdf.end_anchor_tree:
                if isinstance(tail, pd.DataFrame) and not tail.empty:
                    x_vals.extend(list(tail["x"].astype(float)))
                    y_vals.extend(list(tail["y"].astype(float)))
                elif isinstance(tail, dict) and "features" in tail:
                    for f in tail["features"]:
                        props = f.get("properties", f)
                        x_vals.append(float(props.get("x", 0)))
                        y_vals.append(float(props.get("y", 0)))
                elif isinstance(tail, dict):
                    if "x" in tail and "y" in tail:
                        x_vals.append(float(tail["x"]))
                        y_vals.append(float(tail["y"]))

        # 3. all road anchors
        if hasattr(fa.line_gdf, "road_anchor_tree_series"):
            for ra in fa.line_gdf.road_anchor_tree_series:
                if isinstance(ra, pd.DataFrame) and not ra.empty:
                    x_vals.extend(list(ra["x"].astype(float)))
                    y_vals.extend(list(ra["y"].astype(float)))
                elif isinstance(ra, dict) and "features" in ra:
                    for f in ra["features"]:
                        props = f.get("properties", f)
                        x_vals.append(float(props.get("x", 0)))
                        y_vals.append(float(props.get("y", 0)))
                elif isinstance(ra, dict):
                    if "x" in ra and "y" in ra:
                        x_vals.append(float(ra["x"]))
                        y_vals.append(float(ra["y"]))

        # 4. final padded bbox
        if x_vals and y_vals:
            pad = 10.0
            minx, maxx = min(x_vals), max(x_vals)
            miny, maxy = min(y_vals), max(y_vals)
            x_range = (minx - pad, maxx + pad)
            y_range = (miny - pad, maxy + pad)
        else:
            # fallback: use tree extents if something weird happens
            if tree_x and tree_y:
                pad = 10.0
                minx, maxx = min(tree_x), max(tree_x)
                miny, maxy = min(tree_y), max(tree_y)
                x_range = (minx - pad, maxx + pad)
                y_range = (miny - pad, maxy + pad)
            else:
                x_range = (-10, 10)
                y_range = (-10, 10)

        # ----------------------------------------------------------------------------------
        # PRECOMPUTED TREE COLORS (so map.py doesn't have to redo assignment)
        # ----------------------------------------------------------------------------------

        # 0) default -> all green (used when nothing is selected)
        tree_color_default = ["green"] * len(tree_x)

        # 1) union: color each tree by nearest corridor among `indices_to_show`
        if self.indices_to_show:
            tree_colors_by_union = _tree_colors_for_indices(
                self.indices_to_show,
                self.forest_area_3,
                self.dtl_full,
            )
        else:
            tree_colors_by_union = tree_color_default

        # 2) per model row index
        tree_colors_by_model: Dict[int, List[str]] = {}
        # 3) also keyed by the tuple of selected corridor indices (order-sensitive)
        tree_colors_by_selection: Dict[Tuple[int, ...], List[str]] = {}

        for i, res in self.results_df.iterrows():
            # selected real corridor indices for this model row
            sel_real_all = [int(x) for x in res["selected_lines"]]
            # compute colors via the helper
            colors_i = _tree_colors_for_indices(sel_real_all, self.forest_area_3, self.dtl_full)
            tree_colors_by_model[int(i)] = colors_i
            tree_colors_by_selection[tuple(sel_real_all)] = colors_i

        # stable color map of corridors themselves
        color_map = {
            rid: self.palette[j % len(self.palette)]
            for j, rid in enumerate(self.indices_to_show)
        }

        # final map payload
        self.map = dict(
            # Trees
            tree_x=tree_x,
            tree_y=tree_y,
            tree_bhd_cm=tree_bhd_cm,

            # Precomputed tree colors
            tree_color_default=tree_color_default,
            tree_colors_by_union=tree_colors_by_union,
            tree_colors_by_model=tree_colors_by_model,
            tree_colors_by_selection=tree_colors_by_selection,

            # Corridors + anchors + per-corridor stats
            corridors=corridors,
            color_map=color_map,
            palette=self.palette,
            display_lookup=display_lookup,

            # global extents (nice zoom from interface.py logic)
            x_range=x_range,
            y_range=y_range,

            # bookkeeping
            indices_to_show=list(self.indices_to_show),
        )

    def _build_overview_rows(self) -> None:
        """
        Rows for the comparison table (overview / ranking list).
        """
        rows: List[List[Any]] = []

        for i, res in self.results_df.iterrows():
            sel_real = [int(x) for x in res["selected_lines"] if int(x) in self.forest_area_3.line_gdf.index]
            layout = self.layout_by_model[i]

            rows.append([
                i + 1,
                layout.get("Total Cable Corridor Costs (€)"),
                layout.get("Setup and Takedown, Prod. Costs (€)"),
                layout.get("Ecol. Penalty"),
                layout.get("Ergon. Penalty"),
                str([self.real_to_display.get(int(idx), int(idx)) for idx in sel_real])[1:-1],
                layout.get("Max lateral Yarding Distance (m)"),
                layout.get("Average lateral Yarding Distance (m)"),
                int(np.mean(layout["Supports Amount"])) if layout.get("Supports Amount") else 0,
                layout.get("Cost per m3 (€)"),
                layout.get("Volume per Meter (m3/m)"),
            ])

        self.overview_rows = rows

    def selected_rows(self, selected_index: int) -> List[List[str]]:
        """
        Corridor detail rows for chosen optimization result.
        """
        if selected_index < 0 or selected_index >= len(self.results_df):
            return []

        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))

        sel_real = [int(x) for x in self.results_df.iloc[selected_index]["selected_lines"]
                    if int(x) in valid_line_ids]
        layout = self.layout_by_model[selected_index]

        vols = layout.get("Wood Volume per Cable Corridor (m3)", [])
        sup_count = layout.get("Supports Amount", [])
        sup_heights = layout.get("Supports Height (m)", [])
        avg_tree_h = layout.get("Average Tree Height (m)", [])
        max_yard = layout.get("Max Yarding Distance per Cable Corridor (m)", [])
        avg_yard = layout.get("Average Yarding Distance per Cable Corridor (m)", [])

        fa = self.forest_area_3
        subset = fa.line_gdf.loc[fa.line_gdf.index.isin(sel_real)].loc[sel_real]

        rows: List[List[str]] = []
        for i, real_idx in enumerate(sel_real):
            disp_id = self.real_to_display.get(int(real_idx), int(real_idx))

            line_cost = int(subset.loc[real_idx, "line_cost"]) if "line_cost" in subset.columns else 0
            line_length = int(subset.loc[real_idx, "line_length"]) if "line_length" in subset.columns else 0

            vol = int(vols[i]) if i < len(vols) else 0
            s_cnt = int(sup_count[i]) if i < len(sup_count) else 0
            s_hlst = sup_heights[i] if i < len(sup_heights) and isinstance(sup_heights[i], list) else []
            s_hstr = "/" if not s_hlst else ", ".join(str(int(h)) for h in s_hlst)

            avg_h = float(avg_tree_h[i]) if i < len(avg_tree_h) else 0.0
            max_y = int(max_yard[i]) if i < len(max_yard) else 0
            avg_y = int(avg_yard[i]) if i < len(avg_yard) else 0

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
        Tail anchor / end support info per corridor for the chosen optimization result.
        Matches what interface.py shows in the 'Endmast Informationen' table:
        [Seiltrassen Nummer, BHD [cm], Height [m], X-Koordinate, Y-Koordinate]
        """
        import pandas as pd  # just to be safe inside the function

        if selected_index < 0 or selected_index >= len(self.results_df):
            return []

        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))
        sel_real = [
            int(x) for x in self.results_df.iloc[selected_index]["selected_lines"]
            if int(x) in valid_line_ids
        ]

        fa = self.forest_area_3
        subset = fa.line_gdf.loc[fa.line_gdf.index.isin(sel_real)].loc[sel_real]

        out_rows: List[List[str]] = []

        for real_idx, row in subset.iterrows():
            disp_id = self.real_to_display.get(int(real_idx), int(real_idx))

            # prefer end_support_tree, fall back to end_anchor_tree
            ta = getattr(row, "end_support_tree", None)
            if ta is None:
                ta = getattr(row, "end_anchor_tree", None)

            bhd = h = x = y = None

            # Case 1: pandas Series (this is how interface.py accesses it)
            if isinstance(ta, pd.Series):
                # guard against missing keys
                bhd = ta.get("BHD", None)
                h   = ta.get("h",   None)
                x   = ta.get("x",   None)
                y   = ta.get("y",   None)

            # Case 2: single-row DataFrame
            elif isinstance(ta, pd.DataFrame) and not ta.empty:
                first = ta.iloc[0]
                bhd = first.get("BHD", None)
                h   = first.get("h",   None)
                x   = first.get("x",   None)
                y   = first.get("y",   None)

            # Case 3: dict
            elif isinstance(ta, dict):
                # could be plain dict or geojson-like
                if "features" in ta:
                    # geojson-style: take first feature.properties
                    try:
                        props = ta["features"][0].get("properties", ta["features"][0])
                    except Exception:
                        props = {}
                    bhd = props.get("BHD", None)
                    h   = props.get("h",   None)
                    x   = props.get("x",   None)
                    y   = props.get("y",   None)
                else:
                    bhd = ta.get("BHD", None)
                    h   = ta.get("h",   None)
                    x   = ta.get("x",   None)
                    y   = ta.get("y",   None)

            # format / cast like interface.py (ints for BHD/h, rounded floats for coords)
            def _to_int(v):
                try:
                    return int(v)
                except Exception:
                    return None

            def _to_coord(v):
                try:
                    return round(float(v), 2)
                except Exception:
                    return None

            bhd_val = _to_int(bhd)
            h_val   = _to_int(h)
            x_val   = _to_coord(x)
            y_val   = _to_coord(y)

            out_rows.append([
                str(disp_id),
                "" if bhd_val is None else str(bhd_val),
                "" if h_val   is None else str(h_val),
                "" if x_val   is None else str(x_val),
                "" if y_val   is None else str(y_val),
            ])

        return out_rows

    def make_radar_scores(self, axes: List[str]) -> pd.DataFrame:
        """
        Produce radar data frame for spider plot (eco / ergo / cost).
        """
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

        colors = [
            px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, _ in enumerate(scores.index)
        ]
        scores["color"] = [_convert_hex_to_rgba(c) for c in colors]
        scores["fill_color"] = [_convert_hex_to_rgba(c, 0.18) for c in colors]
        scores["raw_eco"] = self.results_df.loc[df.index, "ecological_distances_RNI"]
        scores["raw_ergo"] = self.results_df.loc[df.index, "ergonomics_distances_RNI"]
        scores["raw_cost"] = self.results_df.loc[df.index, "cost_objective_RNI"]

        # triangle area for "bigger is better"
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])

        def _tri_area(row):
            r = np.array([row[axes[0]], row[axes[1]], row[axes[2]]], dtype=float)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            return 0.5 * abs(
                x[0] * y[1] + x[1] * y[2] + x[2] * y[0]
                - y[0] * x[1] - y[1] * x[2] - y[2] * x[0]
            )

        scores["triangle_area"] = scores.apply(_tri_area, axis=1)
        return scores

    def to_string(self, full: bool = False) -> str:
        """
        Lightweight pretty-printer for debugging in notebooks.
        """
        def fmt_any(v) -> str:
            try:
                if isinstance(v, np.ndarray):
                    return f"ndarray(shape={v.shape}, dtype={v.dtype})"
            except Exception:
                pass
            if isinstance(v, pd.DataFrame):
                return f"DataFrame(shape={v.shape}, cols={list(v.columns)})"
            if isinstance(v, pd.Series):
                return f"Series(len={len(v)}, name={v.name})"
            return repr(v)

        lines = []
        lines.append(f"{self.__class__.__name__}" + "{{")
        lines.append("  # Core")
        lines.append(f"  indices_to_show: {self.indices_to_show}")
        lines.append(f"  display_to_real: {self.display_to_real}")
        lines.append(f"  real_to_display: {self.real_to_display}")
        lines.append(f"  dtl_full: {fmt_any(self.dtl_full)}")
        lines.append(f"  dcs_full: {fmt_any(self.dcs_full)}")
        lines.append(f"  palette: {px.colors.qualitative.Plotly}")
        lines.append("\n  # Layouts")
        lines.append("  layout_union: " + fmt_any(self.layout_union))
        lines.append(f"  layout_by_model: dict(len={len(self.layout_by_model)})")
        lines.append("\n  # Map")
        lines.append("  map keys: " + ", ".join(list(self.map.keys())))
        lines.append("\n  # Overview rows")
        lines.append(f"  rows={len(self.overview_rows)}")
        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_string(full=False)


# --------------------------------------------------------------------------------------
# Public convenience wrappers
# --------------------------------------------------------------------------------------

def build_viz_data(forest_area_3, model_list, results_df: pd.DataFrame) -> VizData:
    return VizData(forest_area_3=forest_area_3, model_list=model_list, results_df=results_df)


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
