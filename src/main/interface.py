import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import Button, Layout, Checkbox, HBox, VBox, Label, HTML
from IPython.display import display as nb_display, HTML as NBHTML

from src.main import geometry_operations, plotting_3d


# ---------- Theming helpers ----------

BG_GREY = "rgb(217, 217, 217)"
GREEN_ALT = "rgba(139, 255, 129, 0.12)"  # 12% green
WHITE = "white"


def get_zebra_fill_white_first(n_rows: int, n_cols: int) -> list:
    """Return alternating row colors for tables starting with GREEN, then WHITE, etc."""
    row_colors = [WHITE if i % 2 == 1 else GREEN_ALT for i in range(n_rows)]
    return [row_colors for _ in range(n_cols)]


def _pad_headers(values):
    """Add lightweight visual margin around header titles."""
    def pad(v):
        return f"\u00A0{v}\u00A0"
    return [pad(v) for v in values]


def style_table(table: go.FigureWidget) -> None:
    """Apply consistent styling and zebra stripes to a plotly table."""
    if not table.data:
        return

    tbl = table.data[0]
    if not hasattr(tbl, "cells") or not hasattr(tbl, "header"):
        return

    # Header: WHITE background, padded titles, taller row
    header_vals = getattr(tbl.header, "values", [])
    if header_vals:
        tbl.header.values = _pad_headers(header_vals)

    tbl.header.update(
        fill_color=WHITE,
        line_color="rgba(0, 0, 0, 0)",
        align="center",
        font=dict(color="black", size=12, family="Arial"),
        height=34,
    )

    # Cells styled with zebra pattern (WHITE first)
    values = getattr(tbl.cells, "values", None)
    if not values or len(values) == 0:
        return

    n_rows = len(values[0]) if isinstance(values[0], (list, tuple, np.ndarray, pd.Series)) else 0
    n_cols = len(values)
    tbl.cells.update(
        fill_color=get_zebra_fill_white_first(n_rows, n_cols),
        line_color=[["rgba(0, 0, 0, 0)"] * n_rows for _ in range(n_cols)],
        align="center",
        font=dict(color="black", family="Arial"),
        height=26,
    )


def create_trees_and_lines_traces(forest_area_3, transparent_line, selected_indices=None, display_names=None):
    # create a trace for the trees
    xs, ys = zip(
        *[
            (row.xy[0][0], row.xy[1][0])
            for row in forest_area_3.harvesteable_trees_gdf.geometry
        ]
    )
    trees = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(color="green"),
        name="Trees",
        customdata=forest_area_3.harvesteable_trees_gdf["BHD"],
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>BHD: %{customdata} cm<extra></extra>",
        showlegend=False,
    )

    if selected_indices is None:
        selected_indices = list(forest_area_3.line_gdf.index)

    if display_names is None:
        display_names = selected_indices

    df = forest_area_3.line_gdf.loc[selected_indices]

    # Create traces for each line
    individual_lines = []
    for display, (idx, row) in zip(display_names, df.iterrows()):
        line = row.geometry
        # sample multiple points along the line so hovering works anywhere
        n_points = max(int(line.length // 5) + 2, 20)
        distances = np.linspace(0, line.length, n_points)
        x_coords = []
        y_coords = []
        for d in distances:
            pt = line.interpolate(d)
            x_coords.append(pt.x)
            y_coords.append(pt.y)

        individual_lines.append(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=transparent_line,
                name=str(display),  # display number
                meta=int(idx),      # real index
                legendgroup=str(display),
                showlegend=True,
            )
        )

    return trees, individual_lines


def create_anchor_traces(forest_area_3, transparent_line, color_map, real_to_display, selected_indices=None):
    """Create scatter traces and connecting lines for anchor trees."""
    if selected_indices is None:
        selected_indices = list(forest_area_3.line_gdf.index)

    tail_markers = []
    road_markers = []
    tail_lines = []
    road_lines = []

    for idx in selected_indices:
        row = forest_area_3.line_gdf.loc[idx]

        line = row.geometry
        start_pt = line.coords[0]
        end_pt = line.coords[-1]
        anchor_df = row.end_anchor_tree
        color = color_map[idx]
        display_idx = real_to_display[idx]

        ex = round(float(anchor_df.loc["x"]), 2)
        ey = round(float(anchor_df.loc["y"]), 2)

        tail_markers.append(
            go.Scatter(
                x=[ex],
                y=[ey],
                mode="markers",
                marker=dict(color=color, symbol="triangle-up", size=10),
                showlegend=False,
                name=f"Tail Anchor {display_idx}",
                customdata=[[int(anchor_df.loc["BHD"]), idx]],
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>BHD: %{customdata[0]} cm<extra></extra>",
                meta=int(idx),
                legendgroup=str(display_idx),
            )
        )

        tail_lines.append(
            go.Scatter(
                x=[end_pt[0], ex],
                y=[end_pt[1], ey],
                mode="lines",
                line=dict(color=color, dash="dot", width=1),
                showlegend=False,
                hoverinfo="skip",
                meta=int(idx),
                legendgroup=str(display_idx),
                visible=True,
            )
        )

        road_anchor_df = row.road_anchor_tree_series

        # Ensure road_anchor_df is iterable: include ALL anchors to avoid cropping
        if isinstance(road_anchor_df, pd.DataFrame):
            road_anchor_iter = road_anchor_df.to_dict("records")
        elif isinstance(road_anchor_df, dict) and "features" in road_anchor_df:
            road_anchor_iter = road_anchor_df["features"]
        else:
            road_anchor_iter = []

        for road_anchor in road_anchor_iter:
            properties = road_anchor.get("properties", road_anchor)
            rx = round(float(properties.get("x", 0)), 2)
            ry = round(float(properties.get("y", 0)), 2)

            road_markers.append(
                go.Scatter(
                    x=[rx],
                    y=[ry],
                    mode="markers",
                    marker=dict(color=color, symbol="triangle-down", size=10),
                    showlegend=False,
                    name=f"Road Anchor {display_idx}",
                    customdata=[[int(properties.get("BHD", 0)), idx]],
                    hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>BHD: %{customdata[0]} cm<extra></extra>",
                    meta=int(idx),
                    legendgroup=str(display_idx),
                )
            )

            road_lines.append(
                go.Scatter(
                    x=[start_pt[0], rx],
                    y=[start_pt[1], ry],
                    mode="lines",
                    line=dict(color=color, dash="dot", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                    meta=int(idx),
                    legendgroup=str(display_idx),
                    visible=True,
                )
            )

    return tail_markers, tail_lines, road_markers, road_lines


def _toggle_table(fig: go.FigureWidget, visible: bool, ncols: int = 0):
    """Hide/show the plotly table and optionally display a placeholder message."""
    if fig.data:
        fig.data[0].visible = visible
    if visible:
        fig.update_layout(annotations=[])
    else:
        if ncols and fig.data:
            try:
                fig.data[0].cells.values = [[] for _ in range(ncols)]
            except Exception:
                pass


def update_interactive_based_on_indices(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    layout_overview_table_figure,
    anchor_table_figure,
    road_anchor_table_figure,
    current_indices,
    interactive_layout,
    color_map,
    forest_area_3,
    model_list,
    transparent_line,
    solid_line,
    real_to_display,
    precomputed=None,
):
    """
    Update the interactive layout + tables based on the selected real indices.
    """
    # update the tables (no overview table here) and get metrics once
    updated_layout_costs = update_tables_no_layout(
        current_cable_roads_table_figure,
        current_cable_roads_table,
        anchor_table_figure,
        road_anchor_table_figure,
        current_indices,
        interactive_layout,
        forest_area_3,
        model_list,
        real_to_display,
        precomputed=precomputed,
    )
    # recolor with already-computed volumes for fast hover; hide unselected but keep in legend
    update_line_colors_by_indices(
        current_indices, interactive_layout, color_map, forest_area_3, model_list, real_to_display,
        volumes=updated_layout_costs.get("Wood Volume per Cable Corridor (m3)"), hide_unselected=True
    )


def update_colors_and_tables(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    layout_overview_table_figure,
    anchor_table_figure,
    road_anchor_table_figure,
    current_indices,
    interactive_layout,
    color_map,
    forest_area_3,
    model_list,
    real_to_display,
    precomputed=None,
):
    """Wrapper to update both the colors of the lines and the tables."""
    updated_layout_costs = update_tables(
        current_cable_roads_table_figure,
        current_cable_roads_table,
        layout_overview_table_figure,
        anchor_table_figure,
        road_anchor_table_figure,
        current_indices,
        interactive_layout,
        forest_area_3,
        model_list,
        real_to_display,
        precomputed=precomputed,
    )
    update_line_colors_by_indices(
        current_indices, interactive_layout, color_map, forest_area_3, model_list, real_to_display,
        volumes=updated_layout_costs.get("Wood Volume per Cable Corridor (m3)"),
        hide_unselected=True
    )


def update_tables(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    layout_overview_table_figure,
    anchor_table_figure,
    road_anchor_table_figure,
    current_indices,
    interactive_layout,
    forest_area_3,
    model_list,
    real_to_display,
    precomputed=None,
):
    """
    Update all tables with the new selection.
    """
    # update the dataframe showing the computed costs
    updated_layout_costs = update_layout_overview(current_indices, forest_area_3, model_list, precomputed=precomputed)

    layout_overview_table_figure.data[0].cells.values = [
        updated_layout_costs["Total Cable Corridor Costs (€)"],
        updated_layout_costs["Setup and Takedown, Prod. Costs (€)"],
        updated_layout_costs["Ecol. Penalty"],
        updated_layout_costs["Ergon. Penalty"],
        [[real_to_display.get(int(i), int(i)) for i in current_indices]],  # filtered numbers
        updated_layout_costs["Max lateral Yarding Distance (m)"],
        updated_layout_costs["Average lateral Yarding Distance (m)"],
        int(np.mean(updated_layout_costs["Supports Amount"])) if updated_layout_costs.get("Supports Amount") else 0,
        updated_layout_costs["Cost per m3 (€)"],
        updated_layout_costs["Volume per Meter (m3/m)"],
    ]

    # Current cable roads table (only selected CRs)
    subset = current_cable_roads_table.loc[current_cable_roads_table.index.isin(current_indices)]

    corridor_numbers = [real_to_display.get(int(i), int(i)) for i in subset.index]
    line_costs = subset["line_cost"].values
    line_lengths = subset["line_length"].values

    support_heights = [
        "/" if len(h) == 0 else ", ".join(map(lambda x: str(int(x)), h))
        for h in updated_layout_costs["Supports Height (m)"]
    ]

    # show/hide detail tables by selection
    if len(current_indices) == 0:
        _toggle_table(current_cable_roads_table_figure, False, 9)
        _toggle_table(road_anchor_table_figure, False, 5)
    else:
        _toggle_table(current_cable_roads_table_figure, True)
        _toggle_table(road_anchor_table_figure, True)

    current_cable_roads_table_figure.data[0].cells.values = [
        corridor_numbers,
        line_costs.astype(int),
        line_lengths.astype(int),
        updated_layout_costs["Wood Volume per Cable Corridor (m3)"],
        updated_layout_costs["Supports Amount"],
        support_heights,
        updated_layout_costs["Average Tree Height (m)"],
        updated_layout_costs["Max Yarding Distance per Cable Corridor (m)"],
        updated_layout_costs["Average Yarding Distance per Cable Corridor (m)"],
    ]

    # color the trees by assignment
    if len(updated_layout_costs["Tree to Cable Corridor Assignment"]) == len(forest_area_3.harvesteable_trees_gdf):
        interactive_layout.data[0].marker.color = [
            px.colors.qualitative.Plotly[integer]
            for integer in updated_layout_costs["Tree to Cable Corridor Assignment"]
        ]

    # End mast table
    anchor_table_figure.data[0].cells.values = [
        [real_to_display.get(int(i), int(i)) for i in updated_layout_costs["Corresponding Cable Corridor"]],
        updated_layout_costs["Anchor BHD"],
        updated_layout_costs["Anchor height"],
        updated_layout_costs["Anchor x coordinate"],
        updated_layout_costs["Anchor y coordinate"],
    ]

    # Road anchor table
    road_anchor_table_figure.data[0].cells.values = [
        [real_to_display.get(int(i), int(i)) for i in updated_layout_costs["Corresponding Cable Corridor"]],
        updated_layout_costs["Road Anchor BHD"],
        updated_layout_costs["Road Anchor height"],
        updated_layout_costs["Road Anchor x coordinate"],
        updated_layout_costs["Road Anchor y coordinate"],
    ]

    style_table(current_cable_roads_table_figure)
    style_table(layout_overview_table_figure)
    style_table(anchor_table_figure)
    style_table(road_anchor_table_figure)

    return updated_layout_costs


def update_tables_no_layout(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    anchor_table_figure,
    road_anchor_table_figure,
    current_indices,
    interactive_layout,
    forest_area_3,
    model_list,
    real_to_display,
    precomputed=None,
):
    """Update tables without altering the overview table."""
    updated_layout_costs = update_layout_overview(current_indices, forest_area_3, model_list, precomputed=precomputed)

    subset = current_cable_roads_table.loc[current_cable_roads_table.index.isin(current_indices)]

    corridor_numbers = [real_to_display.get(int(i), int(i)) for i in subset.index]
    line_costs = subset["line_cost"].values
    line_lengths = subset["line_length"].values

    support_heights = [
        "/" if len(h) == 0 else ", ".join(map(lambda x: str(int(x)), h))
        for h in updated_layout_costs["Supports Height (m)"]
    ]

    if len(current_indices) == 0:
        _toggle_table(current_cable_roads_table_figure, False, 9)
        _toggle_table(road_anchor_table_figure, False, 5)
    else:
        _toggle_table(current_cable_roads_table_figure, True)
        _toggle_table(road_anchor_table_figure, True)

    current_cable_roads_table_figure.data[0].cells.values = [
        corridor_numbers,
        line_costs.astype(int),
        line_lengths.astype(int),
        updated_layout_costs["Wood Volume per Cable Corridor (m3)"],
        updated_layout_costs["Supports Amount"],
        support_heights,
        updated_layout_costs["Average Tree Height (m)"],
        updated_layout_costs["Max Yarding Distance per Cable Corridor (m)"],
        updated_layout_costs["Average Yarding Distance per Cable Corridor (m)"],
    ]

    if len(updated_layout_costs["Tree to Cable Corridor Assignment"]) == len(forest_area_3.harvesteable_trees_gdf):
        interactive_layout.data[0].marker.color = [
            px.colors.qualitative.Plotly[i]
            for i in updated_layout_costs["Tree to Cable Corridor Assignment"]
        ]

    anchor_table_figure.data[0].cells.values = [
        [real_to_display.get(int(i), int(i)) for i in updated_layout_costs["Corresponding Cable Corridor"]],
        updated_layout_costs["Anchor BHD"],
        updated_layout_costs["Anchor height"],
        updated_layout_costs["Anchor x coordinate"],
        updated_layout_costs["Anchor y coordinate"],
    ]

    road_anchor_table_figure.data[0].cells.values = [
        [real_to_display.get(int(i), int(i)) for i in updated_layout_costs["Corresponding Cable Corridor"]],
        updated_layout_costs["Road Anchor BHD"],
        updated_layout_costs["Road Anchor height"],
        updated_layout_costs["Road Anchor x coordinate"],
        updated_layout_costs["Road Anchor y coordinate"],
    ]

    style_table(current_cable_roads_table_figure)
    style_table(anchor_table_figure)
    style_table(road_anchor_table_figure)

    return updated_layout_costs


def create_contour_traces(forest_area_3):
    """Create the contour traces for the given forest area at the given resolution"""
    # only select ~200 points, as everything else is too much and crashes
    small_contour_height_gdf = forest_area_3.height_gdf.iloc[::10000]
    # and only get points in a certain range to preserve our frame of reference
    small_contour_height_gdf = small_contour_height_gdf[
        (small_contour_height_gdf.x.between(-130, 20))
        & (small_contour_height_gdf.y.between(-20, 165))
    ]

    # create the traces
    data = go.Contour(
        z=small_contour_height_gdf.elev.values,
        x=small_contour_height_gdf.x.values,
        y=small_contour_height_gdf.y.values,
        opacity=0.3,
        showscale=False,
        hoverinfo="none",
        colorscale="Greys",
        name="Contour",
    )
    return data


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


def update_line_colors_by_indices(
    current_indices,
    interactive_layout,
    color_map,
    forest_area_3=None,
    model_list=None,
    real_to_display=None,
    hide_unselected=False,
    volumes=None,
):
    """Set line colors and hover information for the currently active cable roads.
    If hide_unselected=True, hide non-used corridors from the map but keep them in the legend."""
    order = sorted(current_indices)
    palette = px.colors.qualitative.Plotly

    def cr_colour(idx: int) -> str:
        if idx in order and len(order) > 0:
            return palette[order.index(idx) % len(palette)]
        return "rgba(0,0,0,0.5)"

    current_indices = list(map(int, current_indices))
    color_transparent = "rgba(0, 0, 0, 0.4)"
    default_marker_color = "green" if not current_indices else color_transparent

    # reset traces (but do not fully hide to keep legend by default)
    for trace in interactive_layout.data[2:]:
        idx = getattr(trace, "meta", None)
        if idx is None:
            continue
        # default: demote to legendonly for lines, hide anchors
        if hasattr(trace, "line"):
            trace.line.color = color_transparent
            trace.line.width = 1 if getattr(trace.line, "dash", None) else 0.5
            trace.hovertemplate = None
            if hide_unselected:
                # only CR polylines should become legendonly; keep dotted anchor lines fully hidden
                if getattr(trace.line, "dash", None):
                    trace.visible = False
                else:
                    trace.visible = "legendonly"
            else:
                trace.visible = True
        elif getattr(trace, "mode", "").startswith("markers"):
            trace.marker.color = default_marker_color
            if "Tail Anchor" in trace.name:
                trace.marker.symbol = "triangle-up"
            elif "Road Anchor" in trace.name:
                trace.marker.symbol = "square"
            else:
                trace.marker.symbol = "circle"
            trace.visible = (False if hide_unselected else True)

    # apply colors to active lines + show their anchors
    for indice in current_indices:
        color = cr_colour(indice)
        hover = None
        if volumes is not None and indice in forest_area_3.line_gdf.index:
            length = int(forest_area_3.line_gdf.loc[indice, "line_length"])
            vol = volumes[current_indices.index(indice)]
            disp = real_to_display.get(indice, indice) if real_to_display else indice
            hover = f"Seiltrasse {disp}<br>Seillänge: {length} m<br>Vfm: {vol} m³"

        for trace in interactive_layout.data[2:]:
            if getattr(trace, "meta", None) != indice:
                continue

            if hasattr(trace, "line"):
                trace.line.color = color
                trace.line.width = 1 if getattr(trace.line, "dash", None) else 5
                # show selected lines; dotted anchor lines of selected CRs stay visible
                if getattr(trace.line, "dash", None):
                    trace.visible = True
                else:
                    trace.visible = True
                if hover is not None and not getattr(trace.line, "dash", None):
                    trace.hovertemplate = hover + "<extra></extra>"
            elif getattr(trace, "mode", "").startswith("markers"):
                trace.marker.color = color
                if "Tail Anchor" in trace.name:
                    trace.marker.symbol = "triangle-up"
                elif "Road Anchor" in trace.name:
                    trace.marker.symbol = "square"
                else:
                    trace.marker.symbol = "circle"
                trace.visible = True


def _build_visibility_table_widget(results_df, point_visible):
    """
    Hybrid 'visibility' table:
      • Left: Plotly Table (first 4 columns) styled like the other tables
      • Right: ipywidgets checkbox column ('Anzeigen / Ausblenden') with real interaction
      • A shared external title bar above both so everything aligns cleanly.
    """
    # ---- CSS for this composite widget (scoped via classes we add below) ----
    nb_display(NBHTML(f"""
    <style id="visibility-table-css">
    /* Hide Plotly modebar only for this small table */
    .vis-plot .modebar {{ display:none !important; }}
    /* Title bar */
    .vis-title {{
        font-family: Arial, sans-serif; font-weight: 600;
        padding: 8px 30px 0 30px; color: #000;
    }}
    /* Align the two halves on their top edge */
    .vis-container .widget-hbox {{ align-items: flex-start !important; }}
    /* Right checkbox column layout and zebra */
    .visibility-col {{
        width: 160px;
        margin-left: 8px;
    }}
    .visibility-col .v-header {{
        background: {WHITE};
        text-align: center;
        font-family: Arial, sans-serif;
        font-weight: 600;
        height: 34px; line-height: 34px;
        padding: 0 8px;
        border: 0;
    }}
    .visibility-col .row {{
        display:flex; align-items:center; justify-content:center;
        height: 26px; /* match Plotly cell height */
        padding: 0 8px;
    }}
    .visibility-col .row.even {{ background:{WHITE}; }}
    .visibility-col .row.odd  {{ background:{GREEN_ALT}; }}
    .visibility-col * {{ box-shadow:none !important; border:0 !important; }}
    /* Remove any scrollbar arrow buttons if a notebook theme adds them */
    .visibility-col::-webkit-scrollbar-button {{ display:none; width:0; height:0; }}
    </style>
    """))

    ergonomics_column = (
        "ergonomics_discances_RNI"
        if "ergonomics_discances_RNI" in results_df.columns
        else "ergonomics_distances_RNI"
    )

    # ---- Left side: Plotly table (no internal title/modebar) ----
    headers_left = [
        "Index",
        "Kosten Optimierung",
        "Ergonomische Optimierung",
        "Ökologische Optimierung",
    ]
    index_col = [i + 1 for i in range(len(results_df))]
    kosten_col = [int(x) for x in results_df["cost_objective_RNI"].tolist()]
    ergo_col  = [int(x) for x in results_df[ergonomics_column].tolist()]
    oeko_col  = [int(x) for x in results_df["ecological_distances_RNI"].tolist()]

    fig_left = go.FigureWidget([
        go.Table(
            header=dict(
                values=_pad_headers(headers_left),
                fill_color=WHITE,
                align="center",
                line_color="darkgrey",
                font=dict(color="black", size=12, family="Arial"),
                height=34,
            ),
            # fixed widths so it feels table-like and aligns with the checkbox column
            columnwidth=[60, 120, 150, 150],
            cells=dict(
                values=[index_col, kosten_col, ergo_col, oeko_col],
                align="center",
                height=26,
            ),
        )
    ])
    # remove internal title/margins; set grey background like other tables
    fig_left.update_layout(
        margin=dict(r=30, l=30, t=0, b=30),
        paper_bgcolor=BG_GREY,
        plot_bgcolor=BG_GREY,
    )
    # make row heights deterministic & apply zebra
    style_table(fig_left)
    # mark this figure so our CSS can hide only its modebar
    fig_left.add_class("vis-plot")

    return fig_left


def interactive_cr_selection(
    forest_area_3, model_list, optimization_result_list, results_df):
    """
    Create an interactive cable road layout visualization.
f
    Returns (10 items):
      interactive_layout, current_cable_roads_table_figure, layout_overview_table_figure,
      model_visibility_table, anchor_table_figure, road_anchor_table_figure,
      pareto_frontier, move_left_button, move_right_button, reset_all__CRs_button
    """
    # initialize the current indices list we use to keep track of the selected lines
    current_indices = []  # real indices
    selected_cr = None    # real index of a single selected CR or None

    # define the transparent color for CRs once
    color_transparent = "rgba(0, 0, 0, 0.4)"
    transparent_line = dict(color=color_transparent, width=0.5)
    solid_line = dict(color="black", width=5)

    indices_to_show = sorted({int(i) for row in results_df["selected_lines"] for i in row})
    display_names = list(range(1, len(indices_to_show) + 1))
    display_to_real = dict(zip(display_names, indices_to_show))
    real_to_display = dict(zip(indices_to_show, display_names))

    color_map = {
        idx: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, idx in enumerate(indices_to_show)
    }

    # Precompute distance matrices once for performance
    all_distance_tree_line, all_distance_carriage_support = geometry_operations.compute_distances_facilities_clients(
        forest_area_3.harvesteable_trees_gdf, forest_area_3.line_gdf
    )
    precomputed = (all_distance_tree_line, all_distance_carriage_support)

    # create traces for the lines and trees
    trees, individual_lines = create_trees_and_lines_traces(
        forest_area_3,
        transparent_line,
        selected_indices=indices_to_show,
        display_names=display_names,
    )

    # create the traces for a contour plot
    contour_traces = create_contour_traces(forest_area_3)

    # anchors
    tail_anchors, tail_anchor_lines, road_anchors, road_anchor_lines = create_anchor_traces(
        forest_area_3, transparent_line, color_map, real_to_display, selected_indices=indices_to_show
    )

    # interactive figure
    interactive_layout = go.FigureWidget(
        [trees, contour_traces, *individual_lines, *tail_anchor_lines, *road_anchor_lines, *tail_anchors, *road_anchors]
    )
    # background - make the whole interface look grey (not just corners)
    interactive_layout.update_layout(
        title="Cable Corridor Map",
        width=1200,
        height=900,
        xaxis=dict(title="X (m)"),
        yaxis=dict(title="Y (m)"),
        margin=dict(r=20, l=20, t=20, b=20),
        paper_bgcolor=BG_GREY,
        plot_bgcolor=BG_GREY,
        legend=dict(itemsizing="constant"),
    )

    # determine map extent from ALL line vertices + ALL anchors (tail + road)
    x_vals = []
    y_vals = []

    # Lines
    for line in forest_area_3.line_gdf.geometry:
        xs, ys = line.xy
        x_vals.extend(xs)
        y_vals.extend(ys)

    # All tail anchors (use DataFrames fully)
    for anchors_df in forest_area_3.line_gdf.end_support_tree:
        if isinstance(anchors_df, pd.DataFrame) and not anchors_df.empty:
            x_vals.extend(list(anchors_df["x"].astype(float)))
            y_vals.extend(list(anchors_df["y"].astype(float)))
        elif isinstance(anchors_df, dict) and "features" in anchors_df:
            for f in anchors_df["features"]:
                props = f.get("properties", f)
                x_vals.append(float(props.get("x", 0)))
                y_vals.append(float(props.get("y", 0)))

    # All road anchors
    for anchors_df in forest_area_3.line_gdf.road_anchor_tree_series:
        if isinstance(anchors_df, pd.DataFrame) and not anchors_df.empty:
            x_vals.extend(list(anchors_df["x"].astype(float)))
            y_vals.extend(list(anchors_df["y"].astype(float)))
        elif isinstance(anchors_df, dict) and "features" in anchors_df:
            for f in anchors_df["features"]:
                props = f.get("properties", f)
                x_vals.append(float(props.get("x", 0)))
                y_vals.append(float(props.get("y", 0)))

    if x_vals and y_vals:
        margin = 10
        x_range = [min(x_vals) - margin, max(x_vals) + margin]
        y_range = [min(y_vals) - margin, max(y_vals) + margin]
        interactive_layout.update_xaxes(range=x_range)
        interactive_layout.update_yaxes(range=y_range)

    # current cable roads table
    current_cable_roads_table = forest_area_3.line_gdf[["line_cost", "line_length"]].copy()
    current_cable_roads_table.loc[:, "current_wood_volume"] = pd.Series(dtype="int")
    current_cable_roads_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=_pad_headers([
                        "Seiltrassen Nummer",
                        "Aufbaukosten [€]",
                        "Seillänge [m]",
                        "Vfm pro Seiltrasse [m³]",
                        "Stützbaum Anzahl",
                        "Tragseilhöhe Stütze [m]",
                        "Durchschnittliche Baumhöhe [m]",
                        "Max Zugseillänge [m]",
                        "Durchschnittliche Zugseillänge [m]",
                    ]),
                    fill_color=WHITE,
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial"),
                    height=34,
                ),
                cells=dict(values=[], align="center"),
            )
        ]
    )
    current_cable_roads_table_figure.update_layout(
        title="Aktivierte Seiltrassen",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
        paper_bgcolor=BG_GREY,
        plot_bgcolor=BG_GREY,
    )
    style_table(current_cable_roads_table_figure)
    _toggle_table(current_cable_roads_table_figure, False, 9)

    # layout overview (Vergleich der Seiltrassenmodelle)
    layout_columns = [
        "Gesamt Kosten [€]",
        "Auf- und Abbau Kosten [€]",
        "Ökologische Penalty",
        "Ergonomische Penalty",
        "Verwendete Seillinien",
        "Max Zuzugslänge [m]",
        "Durchschnittliche Zuzugslänge [m]",
        "Durchschnittliche Stützbaum Anzahl",
        "Kosten pro Vfm [€/m³]",
        "Vfm pro m Seillänge [m³/m]",
    ]

    all_layouts = []
    ergonomics_column = (
        "ergonomics_discances_RNI"
        if "ergonomics_discances_RNI" in results_df.columns
        else "ergonomics_distances_RNI"
    )
    for i, row in results_df.iterrows():
        sel = list(map(int, row["selected_lines"]))  # real indices
        layout_data = update_layout_overview(sel, forest_area_3, model_list, precomputed=precomputed)
        avg_supports = int(np.mean(layout_data["Supports Amount"])) if layout_data.get("Supports Amount") else 0
        # show FILTERED (display) numbers in the overview table
        used_lines_display = [real_to_display.get(int(idx), int(idx)) for idx in sel]
        all_layouts.append(
            [
                i + 1,
                layout_data["Total Cable Corridor Costs (€)"],
                layout_data["Setup and Takedown, Prod. Costs (€)"],
                layout_data["Ecol. Penalty"],
                layout_data["Ergon. Penalty"],
                used_lines_display,
                layout_data["Max lateral Yarding Distance (m)"],
                layout_data["Average lateral Yarding Distance (m)"],
                avg_supports,
                layout_data["Cost per m3 (€)"],
                layout_data["Volume per Meter (m3/m)"],
            ]
        )

    layout_overview_df = pd.DataFrame(all_layouts, columns=["Index"] + layout_columns)

    layout_overview_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=_pad_headers(["Index"] + layout_columns),
                    fill_color=WHITE,
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial"),
                    height=34,
                ),
                cells=dict(values=[layout_overview_df[col] for col in layout_overview_df.columns], align="center"),
            )
        ]
    )
    layout_overview_table_figure.update_layout(
        title="Vergleich der Seiltrassenmodelle",
        margin=dict(r=30, l=30, t=30, b=30),
        paper_bgcolor=BG_GREY,
        plot_bgcolor=BG_GREY,
    )
    style_table(layout_overview_table_figure)

    point_visible = [True] * len(results_df)
    model_visibility_table =_build_visibility_table_widget(results_df, point_visible)

    selected_layout_row = None

    def highlight_layout_row(idx):
        """Highlight the selected row in the layout overview table."""
        nonlocal selected_layout_row
        n_rows = len(layout_overview_df)
        n_cols = len(layout_overview_df.columns)

        fill_colors = get_zebra_fill_white_first(n_rows, n_cols)
        if idx is not None:
            for c in range(n_cols):
                fill_colors[c][idx] = "rgba(255, 0, 0, 0.25)"

        layout_overview_table_figure.data[0].cells.fill.color = fill_colors
        layout_overview_table_figure.data[0].cells.line.color = [["rgba(0, 0, 0, 0)"] * n_rows for _ in range(n_cols)]
        layout_overview_table_figure.layout.shapes = []
        selected_layout_row = idx

    highlight_layout_row(None)

    # anchor table
    anchor_columns = [
        "Seiltrassen Nummer",
        "BHD [cm]",
        "Height [m]",
        "X-Koordinate",
        "Y-Koordinate",
    ]
    anchor_df = pd.DataFrame(columns=anchor_columns)
    anchor_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=_pad_headers(anchor_columns),
                    fill_color=WHITE,
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial"),
                    height=34,
                ),
                cells=dict(values=[anchor_df.values], align="center"),
            )
        ]
    )
    anchor_table_figure.update_layout(
        title="Endmast Informationen",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
        paper_bgcolor=BG_GREY,
        plot_bgcolor=BG_GREY,
    )
    style_table(anchor_table_figure)

    road_anchor_df = pd.DataFrame(columns=anchor_columns)
    road_anchor_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=_pad_headers(anchor_columns),
                    fill_color=WHITE,
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial"),
                    height=34,
                ),
                cells=dict(values=[road_anchor_df], align="center"),
            )
        ]
    )
    road_anchor_table_figure.update_layout(
        title="Straßenmast Informationen",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
        paper_bgcolor=BG_GREY,
        plot_bgcolor=BG_GREY,
    )
    style_table(road_anchor_table_figure)
    _toggle_table(road_anchor_table_figure, False, 5)

    # --- Handlers ---
    def layout_table_click(trace, points, selector):
        if points.point_inds:
            index = points.point_inds[0]
            # Read displayed corridor numbers and map back to REAL indices
            displayed = layout_overview_df.iloc[index]["Verwendete Seillinien"]
            corresponding_indices = [display_to_real.get(int(d), int(d)) for d in displayed]
            update_interactive_based_on_indices(
                current_cable_roads_table_figure,
                current_cable_roads_table,
                layout_overview_table_figure,
                anchor_table_figure,
                road_anchor_table_figure,
                corresponding_indices,
                interactive_layout,
                color_map,
                forest_area_3,
                model_list,
                transparent_line,
                solid_line,
                real_to_display,
                precomputed=precomputed,
            )
            highlight_layout_row(index)
            update_selected_marker(index)

    layout_overview_table_figure.data[0].on_click(layout_table_click)

    def plot_pareto_frontier(
        results_df,
        current_indices,
        interactive_layout,
        layout_overview_table_figure,
        current_cable_roads_table_figure,
        current_cable_roads_table,
        forest_area_3,
        transparent_line,
        solid_line,
        model_list,
        real_to_display,
        gamma: float = 0.12,
    ):
        ergonomics_column = (
            "ergonomics_discances_RNI"
            if "ergonomics_discances_RNI" in results_df.columns
            else "ergonomics_distances_RNI"
        )

        x_vals = results_df["ecological_distances_RNI"].to_numpy(dtype=float)
        y_vals = results_df[ergonomics_column].to_numpy(dtype=float)
        z_vals = results_df["cost_objective_RNI"].to_numpy(dtype=float)

        def normalize_and_gamma(arr: np.ndarray) -> np.ndarray:
            arr_min, arr_max = arr.min(), arr.max()
            norm = np.zeros_like(arr) if arr_max == arr_min else (arr - arr_min) / (arr_max - arr_min)
            return np.power(norm, gamma)

        # Map: green=ecology(x), red=ergonomics(y), blue=cost(z)
        g = (normalize_and_gamma(x_vals) * 255).astype(int)
        r = (normalize_and_gamma(y_vals) * 255).astype(int)
        b = (normalize_and_gamma(z_vals) * 255).astype(int)
        colors = [f"rgb({int(rr*0.7 + 255*0.3)}, {int(gg*0.7 + 255*0.3)}, {int(bb*0.7 + 255*0.3)})" for rr, gg, bb in zip(r, g, b)]

        pareto_frontier = go.FigureWidget(
            data=[
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    text=[str(i + 1) for i in range(len(results_df))],
                    mode="markers+text",
                    textposition="middle center",
                    textfont=dict(color="black", size=12),
                    marker=dict(color=colors, size=8, line=dict(color="black", width=3)),
                    showlegend=True,
                ),
                go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode="markers+text",
                    marker=dict(color="red", size=8, line=dict(color="black", width=4)),
                    text=[],
                    textposition="middle center",
                    textfont=dict(color="black", size=12),
                    name="Ausgewählt",
                    showlegend=True,
                ),
                # Legend items explaining the color channels
                go.Scatter3d(x=[x_vals.min()], y=[y_vals.min()], z=[z_vals.min()], mode="markers",
                             marker=dict(color="green", size=8), name="Ökologisch (grün)", showlegend=True),
                go.Scatter3d(x=[x_vals.min()], y=[y_vals.min()], z=[z_vals.min()], mode="markers",
                             marker=dict(color="red", size=8), name="Ergonomisch (rot)", showlegend=True),
                go.Scatter3d(x=[x_vals.min()], y=[y_vals.min()], z=[z_vals.min()], mode="markers",
                             marker=dict(color="blue", size=8), name="Kosten (blau)", showlegend=True),
            ]
        )

        pareto_frontier.update_layout(
            title="Vergleich der Seiltrassenmodelle",
            width=800,
            height=420,
            scene=dict(
                xaxis_title="Ökologische Optimierung",
                yaxis_title="Ergonomische Optimierung",
                zaxis_title="Kosten Optimierung",
                xaxis={"autorange": "reversed"},
                camera=dict(projection=dict(type="orthographic")),
                bgcolor=WHITE,
            ),
            clickmode="event+select",
            scene_camera=dict(
                eye=dict(x=1.7, y=1.7, z=1),
                center=dict(x=0, y=0, z=-0.5),
                projection=dict(type="orthographic"),
            ),
            margin=dict(r=30, l=30, t=30, b=30),
            uniformtext_minsize=12,
            uniformtext_mode="show",
            paper_bgcolor=BG_GREY,
            plot_bgcolor=BG_GREY,
            legend=dict(itemsizing="constant"),
        )

        def selection_fn(trace, points, selector):
            nonlocal current_indices
            if not points.point_inds:
                return
            index = points.point_inds[0]
            # get REAL indices for this layout
            current_indices = list(map(int, results_df.iloc[index]["selected_lines"]))
            update_interactive_based_on_indices(
                current_cable_roads_table_figure,
                current_cable_roads_table,
                layout_overview_table_figure,
                anchor_table_figure,
                road_anchor_table_figure,
                current_indices,
                interactive_layout,
                color_map,
                forest_area_3,
                model_list,
                transparent_line,
                solid_line,
                real_to_display,
                precomputed=precomputed,
            )
            highlight_layout_row(index)
            update_selected_marker(index)

        pareto_frontier.data[0].on_click(selection_fn)
        return pareto_frontier

    # pareto
    pareto_frontier = plot_pareto_frontier(
        results_df,
        current_indices,
        interactive_layout,
        layout_overview_table_figure,
        current_cable_roads_table_figure,
        current_cable_roads_table,
        forest_area_3,
        transparent_line,
        solid_line,
        model_list,
        real_to_display,
        gamma=0.12,
    )

    orig_x_vals = results_df["ecological_distances_RNI"].to_numpy(dtype=float)
    ergonomics_column = (
        "ergonomics_discances_RNI"
        if "ergonomics_discances_RNI" in results_df.columns
        else "ergonomics_distances_RNI"
    )
    orig_y_vals = results_df[ergonomics_column].to_numpy(dtype=float)
    orig_z_vals = results_df["cost_objective_RNI"].to_numpy(dtype=float)
    point_visible = [True] * len(results_df)

    def apply_visibility():
        pareto_frontier.data[0].x = [x if v else None for x, v in zip(orig_x_vals, point_visible)]
        pareto_frontier.data[0].y = [y if v else None for y, v in zip(orig_y_vals, point_visible)]
        pareto_frontier.data[0].z = [z if v else None for z, v in zip(orig_z_vals, point_visible)]
        pareto_frontier.data[0].text = [str(i + 1) if v else "" for i, v in enumerate(point_visible)]
        update_selected_marker(None)

    def update_selected_marker(index):
        """Draw a black marker over the selected pareto point."""
        if index is None:
            pareto_frontier.data[1].x = []
            pareto_frontier.data[1].y = []
            pareto_frontier.data[1].z = []
            pareto_frontier.data[1].text = []
        else:
            pareto_frontier.data[1].x = [pareto_frontier.data[0].x[index]]
            pareto_frontier.data[1].y = [pareto_frontier.data[0].y[index]]
            pareto_frontier.data[1].z = [pareto_frontier.data[0].z[index]]
            pareto_frontier.data[1].text = [str(index + 1)]

    update_selected_marker(None)

    # click lines to build a custom set
    def selection_fn(trace, points, selector):
        nonlocal selected_cr
        nonlocal current_indices
        # since the handler is activated for all lines, test if this one has coordinates, i.e. is the clicked line
        if not points.xs:
            return

        # toggle this line
        if trace.line.color != "rgba(0, 0, 0, 0.4)":
            # currently highlighted -> demote to legendonly when deselecting
            trace.visible = "legendonly"
            trace.line.color = "rgba(0, 0, 0, 0.4)"
            trace.line.width = 0.5
        else:
            # highlight
            trace.visible = True
            trace.line.color = "black"
            trace.line.width = 5
            selected_cr = display_to_real[int(trace.name)]

        # gather all active (non-grey) real indices -> they are visible lines with width >=5
        active_traces = list(
            interactive_layout.select_traces(
                selector=lambda x: (
                    bool(getattr(x, "name", ""))
                    and getattr(x, "line", None) is not None
                    and getattr(x.line, "width", 0) >= 5
                )
            )
        )
        current_indices = [
            display_to_real[int(t.name)]
            for t in active_traces
            if str(t.name).isdigit()
        ]

        # fast recolor + update (hide others but keep in legend)
        updated_layout_costs = update_tables(
            current_cable_roads_table_figure,
            current_cable_roads_table,
            layout_overview_table_figure,
            anchor_table_figure,
            road_anchor_table_figure,
            current_indices,
            interactive_layout,
            forest_area_3,
            model_list,
            real_to_display,
            precomputed=precomputed,
        )
        update_line_colors_by_indices(
            current_indices, interactive_layout, color_map, forest_area_3, model_list, real_to_display,
            volumes=updated_layout_costs.get("Wood Volume per Cable Corridor (m3)"), hide_unselected=True
        )

    # attach handler to all line traces
    for trace in interactive_layout.data[2:]:
        trace.on_click(selection_fn)

    # Buttons
    def set_current_cr(left=False):
        """Cycle the currently selected CR left/right within the active set."""
        nonlocal selected_cr
        nonlocal current_indices

        if selected_cr is None or not current_indices:
            return

        # position within currently active
        index_cr = current_indices.index(selected_cr)

        # make current trace legendonly
        interactive_layout.update_traces(visible="legendonly", selector={"meta": selected_cr})

        # compute neighbor (wrap around the display numbers)
        disp_current = real_to_display[selected_cr]
        new_disp = display_to_real[disp_current - 1 if left else disp_current + 1] if (
            (left and disp_current > 1) or (not left and disp_current < len(display_to_real))
        ) else selected_cr

        selected_cr = new_disp
        current_indices[index_cr] = selected_cr

        # set new trace to solid
        interactive_layout.update_traces(visible=True, selector={"meta": selected_cr})
        interactive_layout.update_traces(line=dict(color="black", width=5), selector={"meta": selected_cr})

        update_colors_and_tables(
            current_cable_roads_table_figure,
            current_cable_roads_table,
            layout_overview_table_figure,
            anchor_table_figure,
            road_anchor_table_figure,
            current_indices,
            interactive_layout,
            color_map,
            forest_area_3,
            model_list,
            real_to_display,
            precomputed=precomputed,
        )

    def move_left_callback(button):
        set_current_cr(left=True)

    def move_right_callback(button):
        set_current_cr(left=False)

    def reset_button_callback(button):
        """Reset selected cable roads and UI."""
        nonlocal current_indices
        nonlocal selected_cr
        nonlocal layout_overview_df

        selected_cr = None
        current_indices = []

        # reset layout (show everyone again)
        update_line_colors_by_indices([], interactive_layout, color_map, forest_area_3, model_list, real_to_display, hide_unselected=False)

        # reset tables
        _toggle_table(current_cable_roads_table_figure, False, 9)
        _toggle_table(road_anchor_table_figure, False, 5)
        layout_overview_table_figure.data[0].cells.values = [layout_overview_df[col] for col in layout_overview_df.columns]

        highlight_layout_row(None)
        update_selected_marker(None)

    def create_buttons():
        """Define and wire up the buttons."""
        move_left_button = Button(description="<-")
        move_right_button = Button(description="->")
        reset_all__CRs_button = Button(
            description="Reset all CRs",
            button_style="danger",
            layout=Layout(width="150px", margin="10px 0 0 auto"),
        )

        move_left_button.on_click(move_left_callback)
        move_right_button.on_click(move_right_callback)
        reset_all__CRs_button.on_click(reset_button_callback)

        return (move_left_button, move_right_button, reset_all__CRs_button)

    buttons = list(create_buttons())

    return (
        interactive_layout,
        current_cable_roads_table_figure,
        layout_overview_table_figure,
        model_visibility_table,   # styled checkbox table (VBox)
        anchor_table_figure,
        road_anchor_table_figure,
        pareto_frontier,
        buttons[0],
        buttons[1],
        buttons[2],
    )