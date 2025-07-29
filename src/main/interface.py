import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from ipywidgets.widgets import Button, Layout

from src.main import geometry_operations, plotting_3d


def get_zebra_fill(n_rows: int, n_cols: int) -> list:
    """Return alternating row colors for tables."""
    row_colors = [
        "rgb(217, 217, 217)" if i % 2 == 1 else "rgba(139, 255, 129, 0.12)"
        for i in range(n_rows)
    ]
    return [row_colors for _ in range(n_cols)]

def style_table(table: go.FigureWidget) -> None:
    """Apply consistent styling and zebra stripes to a plotly table."""

    if not table.data:
        return

    tbl = table.data[0]

    if not hasattr(tbl, "cells") or not hasattr(tbl, "header"):
        return

    values = getattr(tbl.cells, "values", None)

    if not values or len(values) == 0 or len(values[0]) == 0:
        tbl.header.update(
            fill_color="rgb(217, 217, 217)",
            line_color="rgba(0, 0, 0, 0)",
            align="center",
            font=dict(color="black", size=12, family="Arial"),
        )
        return
    
    n_rows = len(values[0])
    n_cols = len(values)
    tbl.header.update(
        fill_color="rgb(217, 217, 217)",
        line_color="rgba(0, 0, 0, 0)",
        align="center",
        font=dict(color = "black", family = "Arial"),
    )

    tbl.cells.update(
        fill_color=get_zebra_fill(n_rows, n_cols),
        line_color=[["rgba(0, 0, 0, 0)"] * n_rows for _ in range(n_cols)],
        align="center",
        font=dict(color = "black", family = "Arial"),
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
                name=str(display),
                meta=int(idx),
                legendgroup=str(display),
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
        # print("\n")
        # print(idx, row)

        line = row.geometry
        start_pt = line.coords[0]
        end_pt = line.coords[-1]

        print(row["Cable Road Object"])

        anchor_df = row.tree_anchor_support_trees

        if isinstance(anchor_df, dict) and "features" in anchor_df:
            anchor_iter = anchor_df["features"]
        elif isinstance(anchor_df, pd.DataFrame):
            anchor_iter =anchor_df.to_dict('records')
        else:
            anchor_iter = []

        color = color_map[idx]
        display_idx = real_to_display[idx]

        for anchor in anchor_iter:
            # print("Anchor")
            print(anchor)
            properties = anchor.get("properties", anchor)
            ex = round(properties["x"], 2)
            ey = round(properties["y"], 2)

            tail_markers.append(
                go.Scatter(
                    x=[ex],
                    y=[ey],
                    mode="markers",
                    marker=dict(color=color, symbol="triangle-up", size=10),
                    showlegend=False,
                    name=f"Tail Anchor {display_idx}",
                    customdata=[[int(properties["BHD"]), idx]],
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

        # Ensure road_anchor_df is a DataFrame
        if isinstance(road_anchor_df, dict) and "features" in road_anchor_df:
            road_anchor_iter = road_anchor_df["features"]
        elif isinstance(road_anchor_df, pd.DataFrame):
            road_anchor_iter = road_anchor_df.to_dict('records')
        else:
            road_anchor_iter = []

        for road_anchor in road_anchor_iter:
            properties = road_anchor.get("properties", road_anchor)
            rx = round(properties["x"], 2)
            ry = round(properties["y"], 2)

            road_markers.append(
                go.Scatter(
                    x=[rx],
                    y=[ry],
                    mode="markers",
                    marker=dict(color=color, symbol="triangle-down", size=10),
                    showlegend=False,
                    name=f"Road Anchor {display_idx}",
                    customdata=[[int(properties["BHD"]), idx]],
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
):
    """
    Function to update the interactive layout based on the current indices as well as the corresponding tables
    """

    # first set all traces to lightgrey and hide them
    interactive_layout.update_traces(line=transparent_line)

    # and update the list of current indices to the new ones as well as the layout
    update_tables_no_layout(
        current_cable_roads_table_figure,
        current_cable_roads_table,
        anchor_table_figure,
        road_anchor_table_figure,
        current_indices,
        interactive_layout,
        forest_area_3,
        model_list,
    )
    update_line_colors_by_indices(current_indices, interactive_layout, color_map, forest_area_3, model_list, hide_unselected=True)


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
):
    """
    Wrapper function to update both the colors of the lines and the tables
    """
    update_tables(
        current_cable_roads_table_figure,
        current_cable_roads_table,
        layout_overview_table_figure,
        anchor_table_figure,
        road_anchor_table_figure,
        current_indices,
        interactive_layout,
        forest_area_3,
        model_list,
    )
    update_line_colors_by_indices(current_indices, interactive_layout, color_map, forest_area_3, model_list, hide_unselected=True)


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
):
    """
    Function to update both tables with the new current_indices based on the selected lines
    """

    # and update the dataframe showing the computed costs
    updated_layout_costs = update_layout_overview(
        current_indices, forest_area_3, model_list
    )

    layout_overview_table_figure.data[0].cells.values = [
        updated_layout_costs["Total Cable Corridor Costs (€)"],
        updated_layout_costs["Setup and Takedown, Prod. Costs (€)"],
        updated_layout_costs["Ecol. Penalty"],
        updated_layout_costs["Ergon. Penalty"],
        [current_indices],
        updated_layout_costs["Max lateral Yarding Distance (m)"],
        updated_layout_costs["Average lateral Yarding Distance (m)"],
        updated_layout_costs["Cost per m3 (€)"],
        updated_layout_costs["Volume per Meter (m3/m)"],
    ]

    # set the current_cable_roads_table dataframe rows to show only these CRs
    subset = current_cable_roads_table.loc[
        current_cable_roads_table.index.isin(current_indices)
    ]

    corridor_numbers = subset.index.astype(int)
    line_costs = subset["line_cost"].values
    line_lengths = subset["line_length"].values

    support_heights = [
        "/" if len(h) == 0 else ", ".join(map(lambda x: str(int(x)), h))
        for h in updated_layout_costs["Supports Height (m)"]
    ]

    current_cable_roads_table_figure.data[0].cells.values = [
        corridor_numbers,
        line_costs.astype(int),
        line_lengths.astype(int),
        updated_layout_costs["Wood Volume per Cable Corridor (m3)"],
        updated_layout_costs["Supports Amount"],
        support_heights,
        updated_layout_costs["Average Tree Height (m)"],
        updated_layout_costs["Max Yarding Distance per Cable Corridor (m)"],
        updated_layout_costs["Average Yarding Distance per Cable Corridor (m)"]
    ]

    # as well as the colour of the corresponding trees
    interactive_layout.data[0].marker.color = [
        px.colors.qualitative.Plotly[integer]
        for integer in updated_layout_costs["Tree to Cable Corridor Assignment"]
    ]

    # update the anchor table
    anchor_table_figure.data[0].cells.values = [
        updated_layout_costs["Corresponding Cable Corridor"],
        updated_layout_costs["Anchor BHD"],
        updated_layout_costs["Anchor height"],
        updated_layout_costs["Anchor x coordinate"],
        updated_layout_costs["Anchor y coordinate"],
    ]

    road_anchor_table_figure.data[0].cells.values = [
        updated_layout_costs["Corresponding Cable Corridor"],
        updated_layout_costs["Road Anchor BHD"],
        updated_layout_costs["Road Anchor height"],
        updated_layout_costs["Road Anchor x coordinate"],
        updated_layout_costs["Road Anchor y coordinate"],
    ]

    style_table(current_cable_roads_table_figure)
    style_table(layout_overview_table_figure)
    style_table(anchor_table_figure)
    style_table(road_anchor_table_figure)

def update_tables_no_layout(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    anchor_table_figure,
    road_anchor_table_figure,
    current_indices,
    interactive_layout,
    forest_area_3,
    model_list,
):
    """Update tables without altering the overview table."""

    updated_layout_costs = update_layout_overview(current_indices, forest_area_3, model_list)

    subset = current_cable_roads_table.loc[
        current_cable_roads_table.index.isin(current_indices)
    ]

    corridor_numbers = subset.index.astype(int)
    line_costs = subset["line_cost"].values
    line_lengths = subset["line_length"].values

    support_heights = [
        "/" if len(h) == 0 else ", ".join(map(lambda x: str(int(x)), h))
        for h in updated_layout_costs["Supports Height (m)"]
    ]

    current_cable_roads_table_figure.data[0].cells.values = [
        corridor_numbers,
        line_costs.astype(int),
        line_lengths.astype(int),
        updated_layout_costs["Wood Volume per Cable Corridor (m3)"],
        updated_layout_costs["Supports Amount"],
        support_heights,
        updated_layout_costs["Average Tree Height (m)"],
        updated_layout_costs["Max Yarding Distance per Cable Corridor (m)"],
        updated_layout_costs["Average Yarding Distance per Cable Corridor (m)"]
    ]

    interactive_layout.data[0].marker.color = [
        px.colors.qualitative.Plotly[i]
        for i in updated_layout_costs["Tree to Cable Corridor Assignment"]
    ]

    anchor_table_figure.data[0].cells.values = [
        updated_layout_costs["Corresponding Cable Corridor"],
        updated_layout_costs["Anchor BHD"],
        updated_layout_costs["Anchor height"],
        updated_layout_costs["Anchor x coordinate"],
        updated_layout_costs["Anchor y coordinate"],
    ]

    road_anchor_table_figure.data[0].cells.values = [
        updated_layout_costs["Corresponding Cable Corridor"],
        updated_layout_costs["Road Anchor BHD"],
        updated_layout_costs["Road Anchor height"],
        updated_layout_costs["Road Anchor x coordinate"],
        updated_layout_costs["Road Anchor y coordinate"],
    ]

    style_table(current_cable_roads_table_figure)
    style_table(anchor_table_figure)
    style_table(road_anchor_table_figure)


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


def update_layout_overview(indices, forest_area_3, model_list) -> dict:
    """
    Function to update the cost dataframe with the updated costs for the new configuration based on the selected lines
    Returns distance_trees_to_lines, productivity_cost_overall, line_cost, total_cost, tree_to_line_assignment
    """

    rot_line_gdf = forest_area_3.line_gdf[forest_area_3.line_gdf.index.isin(indices)]

    # Create a matrix with the distance between every tree and line and the distance between the support (beginning of the CR) and the carriage (cloests point on the CR to the tree)
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
        distance_trees_to_lines_sum = sum(distance_trees_to_selected_lines)
    except:
        tree_to_line_assignment = [1 for i in range(len(distance_tree_line))]
        distance_trees_to_lines_sum = sum(distance_tree_line)
        distance_trees_to_selected_lines = []

    # compute the productivity cost
    selected_prod_cost = model_list[0].productivity_cost[:, indices]
    productivity_cost_overall = 0
    for index, val in enumerate(tree_to_line_assignment):
        productivity_cost_overall += selected_prod_cost[index][val]

    # sum of wood volume per CR
    # here we need to compute the sum of wood per tree to line assignment to return this for the CR table
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(len(rot_line_gdf))
    ]
    wood_volume_per_cr = [
        int(
            sum(
                forest_area_3.harvesteable_trees_gdf.iloc[grouped_indices][
                    "cubic_volume"
                ]
            )
        )
        for grouped_indices in grouped_class_indices
    ]

    # get the average tree size per CR
    average_tree_size_per_cr = [
        round(
            sum(forest_area_3.harvesteable_trees_gdf.iloc[grouped_indices]["h"])
            / len(grouped_indices),
            2,
        )
        for grouped_indices in grouped_class_indices
    ]

    # get the height and amount of supports
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

    # compute yarding distances per cable corridor
    max_yarding_distance_per_cr = []
    average_yarding_distance_per_cr = []
    for line_idx, grouped_indices in enumerate(grouped_class_indices):
        if len(grouped_indices) == 0:
            max_yarding_distance_per_cr.append(0)
            average_yarding_distance_per_cr.append(0)
        else:
            dists = distance_carriage_support[grouped_indices, line_idx]
            max_yarding_distance_per_cr.append(int(max(dists)))
            average_yarding_distance_per_cr.append(int(np.mean(dists)))

    # get the tail spar anchor
    endmast_height_list = []
    endmast_BHD_list = []
    endmast_max_holding_force_list = []
    endmast_x_list = []
    endmast_y_list = []
    # get information about the Endmast - height, BHD,
    for tree_anchor_support_tree in rot_line_gdf.tree_anchor_support_trees:
        endmast = tree_anchor_support_tree.iloc[0]
        endmast_height_list.append(int(endmast["h"]))
        endmast_BHD_list.append(int(endmast["BHD"]))
        endmast_max_holding_force_list.append(int(endmast["max_holding_force"]))
        endmast_x_list.append(round(endmast["x"], 2))
        endmast_y_list.append(round(endmast["y"], 2))

    road_anchor_height_list = []
    road_anchor_BHD_list = []
    road_anchor_max_holding_force_list = []
    road_anchor_x_list = []
    road_anchor_y_list = []
    # and do the same for the road side anchor
    for each_road_anchor in rot_line_gdf.road_anchor_tree_series:
        road_anchor = each_road_anchor.sample(n=1)
        # road_anchor = each_road_anchor.iloc[
        #     0  # each_road_anchor["BHD"].idxmax()
        # ]  # choose the anchor with the largest BHD
        road_anchor_height_list.append(int(road_anchor["h"]))
        road_anchor_BHD_list.append(int(road_anchor["BHD"]))
        road_anchor_max_holding_force_list.append(int(road_anchor["max_holding_force"]))
        road_anchor_x_list.append(round(road_anchor["x"], 2))
        road_anchor_y_list.append(round(road_anchor["y"], 2))

    # get the max and average yarding distance
    max_yarding_distance = max(distance_trees_to_selected_lines)
    average_yarding_distance = np.mean(distance_trees_to_selected_lines)

    line_cost = sum(rot_line_gdf["line_cost"])

    # total cost = # get the total cable road costs
    total_cable_road_costs = line_cost + productivity_cost_overall

    cost_per_m3 = total_cable_road_costs / sum(wood_volume_per_cr)

    # calulate the environmental impact of each line beyond 10m lateral distance
    ecological_penalty_threshold = 10
    ecological_penalty_lateral_distances = np.where(
        distance_tree_line > ecological_penalty_threshold,
        distance_tree_line - ecological_penalty_threshold,
        0,
    )

    sum_eco_distances = sum(
        [
            ecological_penalty_lateral_distances[j][i]
            for i, j in zip(
                tree_to_line_assignment,
                range(len(ecological_penalty_lateral_distances)),
            )
        ]
    )

    # double all distances greater than penalty_treshold
    ergonomics_penalty_treshold = 15
    ergonomic_penalty_lateral_distances = np.where(
        distance_tree_line > ergonomics_penalty_treshold,
        (distance_tree_line - ergonomics_penalty_treshold) * 2,
        0,
    )
    sum_ergo_distances = sum(
        [
            ergonomic_penalty_lateral_distances[j][i]
            for i, j in zip(
                tree_to_line_assignment,
                range(len(ergonomic_penalty_lateral_distances)),
            )
        ]
    )

    # wood volume per running meter of cable road
    # get the total length of the cable road
    total_cable_road_length = sum(rot_line_gdf["line_length"])
    volume_per_running_meter = total_cable_road_length / sum(wood_volume_per_cr)

    # return a dict of the results and convert all results to ints for readability
    return {
        "Wood Volume per Cable Corridor (m3)": wood_volume_per_cr,
        "Total Cable Corridor Costs (€)": int(total_cable_road_costs),
        "Setup and Takedown, Prod. Costs (€)": f"{int(line_cost)}, {int(productivity_cost_overall)}",
        "Ecol. Penalty": int(sum_eco_distances),
        "Ergon. Penalty": int(sum_ergo_distances),
        "Tree to Cable Corridor Assignment": tree_to_line_assignment,
        "Supports Height (m)": supports_height,
        "Supports Amount": supports_amount,
        "Max lateral Yarding Distance (m)": int(max_yarding_distance),
        "Average lateral Yarding Distance (m)": int(average_yarding_distance),
        "Cost per m3 (€)": round(cost_per_m3, 2),
        "Average Tree Height (m)": average_tree_size_per_cr,
        "Volume per Meter (m3/m)": round(volume_per_running_meter, 2),
        "Max Yarding Distance per Cable Corridor (m)": max_yarding_distance_per_cr,
        "Average Yarding Distance per Cable Corridor (m)": average_yarding_distance_per_cr,
        "Anchor height": endmast_height_list,
        "Anchor BHD": endmast_BHD_list,
        "Anchor max holding force": endmast_max_holding_force_list,
        "Anchor x coordinate": endmast_x_list,
        "Anchor y coordinate": endmast_y_list,
        "Corresponding Cable Corridor": indices,
        "Road Anchor height": road_anchor_height_list,
        "Road Anchor BHD": road_anchor_BHD_list,
        "Road Anchor max holding force": road_anchor_max_holding_force_list,
        "Road Anchor x coordinate": road_anchor_x_list,
        "Road Anchor y coordinate": road_anchor_y_list,
        "Road Anchor Angle of Attack": rot_line_gdf[
            "angle_between_start_support_and_cr"
        ],
        "Tail Anchor Angle of Attack": rot_line_gdf["angle_between_end_support_and_cr"],
    }


def update_line_colors_by_indices(
        current_indices, 
        interactive_layout, 
        color_map, 
        forest_area_3=None, 
        model_list=None,
        hide_unselected=True):
    """Set line colors and hover information for the currently active cable roads."""

    order = sorted(current_indices)
    palette = px.colors.qualitative.Plotly
    def cr_colour(idx: int) -> str:
        return palette[order.index(idx) % len(palette)]

    current_indices = list(map(int, current_indices))
    color_transparent = "rgba(0, 0, 0, 0.4)"
    default_marker_color = "green" if not current_indices else color_transparent

    volumes = None
    if current_indices and forest_area_3 is not None and model_list is not None:
        layout_data = update_layout_overview(current_indices, forest_area_3, model_list)
        volumes = layout_data["Wood Volume per Cable Corridor (m3)"]

    #reset traces to grey and show tem
    for trace in interactive_layout.data[2:]:
        idx = getattr(trace, "meta", None)
        if idx is None:
            continue
        if hasattr(trace, "line"):
            trace.line.color = color_transparent
            trace.line.width = 1 if getattr(trace.line, "dash", None) else 0.5
            trace.hovertemplate = None
        elif trace.mode.startswith("markers"):
            trace.marker.color = default_marker_color
            if "Tail Anchor" in trace.name:
                trace.marker.symbol = "triangle-up"
            elif "Road Anchor" in trace.name:
                trace.marker.symbol = "square"
            else:
                trace.marker.symbol = "circle"
        
        if hide_unselected:
            trace.visible = "legendonly"
        else:
            trace.visible = True

    # apply colors to active lines
    for indice in current_indices:
        color = cr_colour(indice)
        hover = None
        if volumes is not None and indice in forest_area_3.line_gdf.index:
            length = int(forest_area_3.line_gdf.loc[indice, "line_length"])
            vol = volumes[current_indices.index(indice)]
            hover = f"Index. {indice}<br>Cable Length: {length} m<br>Wood Volume: {vol} m3"

        for trace in interactive_layout.data[2:]:
            if getattr(trace, "meta", None) != indice:
                continue

            if hasattr(trace, "line"):
                if getattr(trace.line, "dash", None):
                    trace.line.color = color
                    trace.line.width = 1 
                else:
                    trace.line.color = color
                    trace.line.width = 5
                trace.visible = True

            elif trace.mode.startswith("markers"):
                trace.marker.color = color
                if "Tail Anchor" in trace.name:
                    trace.marker.symbol = "triangle-up"
                elif "Road Anchor" in trace.name:
                    trace.marker.symbol = "square"
                else:
                    trace.marker.symbol = "circle"
                trace.visible = True


def interactive_cr_selection(
    forest_area_3, model_list, optimization_result_list, results_df):
    """
    Create an interactive cable road layout visualization.

    Parameters:
    - forest_area_3: GeoDataFrame, input forest area data
    - model_list: List, list of models
    - optimization_result_list: List, list of optimization results
    - results_df: DataFrame, optimization results DataFrame

    Returns:
    - Tuple containing four FigureWidgets: interactive_layout, current_cable_roads_table_figure,
      layout_overview_table_figure, pareto_frontier
    """
    # initialize the current indices list we use to keep track of the selected lines
    current_indices = []

    # initialize the selected cr to none
    selected_cr = 0

    # define the transparent color for CRs once
    color_transparent = "rgba(0, 0, 0, 0.4)"
    transparent_line = dict(color=color_transparent, width=0.5)
    solid_line = dict(color="black", width=5)

    indices_to_show = sorted(
        {int(i) for row in results_df["selected_lines"] for i in row}
    )

    display_names = list(range(1, len(indices_to_show) + 1))
    display_to_real = dict(zip(display_names, indices_to_show))
    real_to_display = dict(zip(indices_to_show, display_names))

    color_map = {
        idx: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, idx in enumerate(indices_to_show)
    }

    # create traces for the lines and trees
    trees, individual_lines = create_trees_and_lines_traces(
        forest_area_3,
        transparent_line,
        selected_indices=indices_to_show,
        display_names=display_names,
    )

    # create the traces for a contour plot
    contour_traces = create_contour_traces(forest_area_3)

    # create a figure from all individual scatter lines
    tail_anchors, road_anchors, tail_anchor_lines, road_anchor_lines = create_anchor_traces(
        forest_area_3, 
        transparent_line, 
        color_map, 
        real_to_display, 
        selected_indices=indices_to_show
    )

    # create a figure from all individual scatter lines and anchors lines
    interactive_layout = go.FigureWidget(
        [trees, contour_traces, *individual_lines, *tail_anchor_lines, *road_anchor_lines, *tail_anchors, *road_anchors]
    )
    update_line_colors_by_indices([], interactive_layout, color_map, forest_area_3, model_list, hide_unselected=False)

    # determine map extent so all anchors fit in view
    x_vals = []
    y_vals = []

    for line in forest_area_3.line_gdf.geometry:
        xs, ys = line.xy
        x_vals.extend(xs)
        y_vals.extend(ys)

    for anchors in forest_area_3.line_gdf.tree_anchor_support_trees:
        if getattr(anchors, "empty", False):
            continue
        if isinstance(anchors, pd.DataFrame):
            records = anchors.to_dict(orient="records")
        else:
            records = [anchors]
        for record in records:
            x_vals.append(float(record["x"]))
            y_vals.append(float(record["y"]))

    for anchors in forest_area_3.line_gdf.road_anchor_tree_series:
        if getattr(anchors, "empty", False):
            continue
        if hasattr(anchors, "sample"):
            record = anchors.iloc[0]
        elif isinstance(anchors, (list, tuple)):
            record = anchors[0]
        else:
            records = anchors
        x_vals.append(float(record["x"]))
        y_vals.append(float(record["y"]))

    margin = 10
    x_range = [min(x_vals) - margin, max(x_vals) + margin]
    y_range = [min(y_vals) - margin, max(y_vals) + margin]
        
    interactive_layout.update_layout(
        title="Cable Corridor Map",
        width=1200,
        height=900,
        xaxis=dict(title="X (m)", range=x_range),
        yaxis=dict(title="Y (m)", range=y_range),
        margin=dict(r=0, l=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # create a dataframe and push it to a figurewidget to display details about our selected lines
    current_cable_roads_table = forest_area_3.line_gdf[
        ["line_cost", "line_length"]
    ].copy()
    # current_cable_roads_table["current_wood_volume"] = pd.Series(dtype="int")
    current_cable_roads_table.loc[:, "current_wood_volume"] = pd.Series(dtype="int")
    current_cable_roads_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=[
                        "Seiltrassen Nummer",
                        "Aufbaukosten [€]",
                        "Seillänge [m]",
                        "Vfm pro Seiltrasse [m³]",
                        "Stützbaum Anzahl",
                        "Tragseilhöhe Stütze [m]",
                        "Durchschnittliche Baumhöhe [m]",
                        "Max Zugseillänge [m]",
                        "Durchschnittliche Zugseillänge [m]",
                    ],
                    fill_color="rgb(217, 217, 217)",
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size = 12, family = "Arial")
                ),
                cells=dict(values=[], align="center"),
            )
        ]
    )
    current_cable_roads_table_figure.update_layout(
        title="Aktivierte Seiltrassen",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
    )

    style_table(current_cable_roads_table_figure)

    # and for the layout overview
    layout_columns = [
        "Total Layout Costs (€)",
        "Setup and Takedown, Prod. Costs (€)",
        "Ecol. Penalty",
        "Ergon. Penalty",
        "Selected Cable Corridors",
        "Max lateral Yarding Distance (m)",
        "Average lateral Yarding Distance (m)",
        "Cost per m3 (€/m)",
        "Volume per Meter (m3/m)",
    ]

    all_layouts = []
    for i, row in results_df.iterrows():
        layout_data = update_layout_overview(row["selected_lines"], forest_area_3, model_list)
        all_layouts.append([
            i + 1,
            layout_data["Total Cable Corridor Costs (€)"],
            layout_data["Setup and Takedown, Prod. Costs (€)"],
            layout_data["Ecol. Penalty"],
            layout_data["Ergon. Penalty"],
            layout_data["Corresponding Cable Corridor"],
            layout_data["Max lateral Yarding Distance (m)"],
            layout_data["Average lateral Yarding Distance (m)"],
            layout_data["Cost per m3 (€)"],
            layout_data["Volume per Meter (m3/m)"],
        ])

    layout_overview_df = pd.DataFrame(
        all_layouts,
        columns=["Index"] + layout_columns,
    )

    layout_overview_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=["Index"] + layout_columns,
                    fill_color="rgb(217, 217, 217)",
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial")
                ),
                cells=dict(values=[layout_overview_df[col] for col in layout_overview_df.columns], align="center"),
            )
        ]
    )
    layout_overview_table_figure.update_layout(
        title="Current Cable Corridor Layout Overview",
        height=300,
        margin=dict(r=30, l=30, t=30, b=30),
    )
    style_table(layout_overview_table_figure)

    selected_layout_row = None

    def highlight_layout_row(idx):
        """Highlight the selected row in the layout overview table."""

        nonlocal selected_layout_row
        n_rows = len(layout_overview_df)
        n_cols = len(layout_overview_df.columns)

        fill_colors = get_zebra_fill(n_rows, n_cols)
        if idx is not None:
            for c in range(n_cols):
                fill_colors[c][idx] = "rgba(255, 0, 0, 0.3)"

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
                    values=anchor_columns,
                    fill_color="rgb(217, 217, 217)",
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial")
                ),
                cells=dict(values=[anchor_df.values], align="center")
            )
        ]
    )
    anchor_table_figure.update_layout(
        title="Endmast Informationen",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
    )
    style_table(anchor_table_figure)

    road_anchor_df = pd.DataFrame(columns=anchor_columns)
    road_anchor_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=anchor_columns,
                    fill_color="rgb(217, 217, 217)",
                    align="center",
                    line_color="darkgrey",
                    font=dict(color="black", size=12, family="Arial"),
                ),
                cells=dict(values=[road_anchor_df], align="center"),
            )
        ]
    )
    road_anchor_table_figure.update_layout(
        title="Endmast Informationen",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
    )
    style_table(road_anchor_table_figure)

    def layout_table_click(trace, points, selector):
        if points.point_inds:
            index = points.point_inds[0]
            corresponding_indices = layout_overview_df.iloc[index]["Selected Cable Corridors"]
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
            return np.power(np.zeros_like(arr) if arr_max == arr_min else (arr - arr_min) / (arr_max - arr_min), gamma)
        
        r = (normalize_and_gamma(x_vals) * 255).astype(int)
        g = (normalize_and_gamma(y_vals) * 255).astype(int)
        b = (normalize_and_gamma(z_vals) * 255).astype(int)
        colors = [f"rgb({int(c_r*0.7 + 255*0.3)}, {int(c_g*0.7 + 255*0.3)}, {int(c_b*0.7 + 255*0.3)})" for c_r, c_g, c_b in zip(r, g, b)]

        pareto_frontier = go.FigureWidget(
            data = [
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    text=[str(i + 1) for i in range(len(results_df))],
                    mode="markers+text",
                    textposition="middle center",
                    textfont=dict(color="black", size=12),
                    marker=dict(color=colors, size=8, line=dict(color="black", width=3)),
                    name="solutions",
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
                    name="selected",
                ),
            ]
        )

        pareto_frontier.update_layout(
            title="""Pareto Frontier of Optimal Solutions""",
            width=800,
            height=400,
            scene=dict(
                xaxis_title="Ecological Optimality",
                yaxis_title="Ergonomics Optimality",
                zaxis_title="Cost Optimality",
                xaxis={"autorange": "reversed"},
                camera=dict(projection=dict(type="orthographic"))
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
        )

        def selection_fn(trace, points, selector):
            nonlocal current_indices

            # print("pareto frontier clicked", current_indices)
            # get index of this point in the trace
            index = points.point_inds[0]

            # get the corresponding list of activated cable rows from the dataframe
            current_indices = results_df.iloc[index]["selected_lines"]
            # print("current indices as per df", current_indices)
            # print(type(current_indices))

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
            )
            highlight_layout_row(index)
            update_selected_marker(index)

        pareto_frontier.data[0].on_click(selection_fn)
        return pareto_frontier

    # get the pareto frontier as 3d scatter plot
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
        gamma=0.12,
    )

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

    # create the onclick function to select new CRs
    def selection_fn(trace, points, selector):
        # since the handler is activated for all lines, test if this one has coordinates, ie. is the clicked line
        if points.xs:
            # set the selected cr to none by default
            nonlocal selected_cr
            selected_cr = None

            # deactive this if it is active
            if trace.line.color != color_transparent:
                interactive_layout.update_traces(
                    line=transparent_line,
                    selector={"name": trace.name},
                )

            # and if this is the clicked line and its not yet activated, turn it black
            elif trace.line.color == color_transparent:
                # update this trace to turn black
                interactive_layout.update_traces(
                    line=solid_line,
                    selector={"name": trace.name},
                )

                # and set the selected cr to the name of the trace if it is a new one
                selected_cr = display_to_real[int(trace.name)]

            # get all active traces  - ie those which are not lightgrey. very heavy-handed expression, but it works
            active_traces = list(
                interactive_layout.select_traces(
                    selector=lambda x: (
                        bool(getattr(x, "name", ""))
                        and x.line.color
                        and x.line.color != color_transparent
                        and x.name != "Contour"
                    )
                )
            )

            nonlocal current_indices
            current_indices = [
                display_to_real[int(trace.name)] 
                for trace in active_traces 
                if str(trace.name).isdigit()
            ]

            # color the traces
            # we set the color of the lines in the current indices in consecutive order by choosing corresponding colors from the colorway
            update_line_colors_by_indices(current_indices, interactive_layout, color_map, forest_area_3, model_list, hide_unselected=True)

            # update the tables accordingly
            update_tables(
                current_cable_roads_table_figure,
                current_cable_roads_table,
                layout_overview_table_figure,
                anchor_table_figure,
                road_anchor_table_figure,
                current_indices,
                interactive_layout,
                forest_area_3,
                model_list,
            )

    # add the onclick function to all line traces
    for trace in interactive_layout.data[2:]:
        trace.on_click(selection_fn)

    # add the custom buttons
    def set_current_cr(left=False):
        """
        Function to set the currently selected cr to the next one
        Refers to the nonlocal variables selected_cr and current_indices
        First we get the index of the cr, then we set the current cr to lightgrey, then we increment/decrement the cr, then we set the new cr to black
        And finally we update the tables and the layout
        """
        nonlocal selected_cr
        nonlocal current_indices

        # if there are no current indices, return
        if selected_cr is None:
            return

        # get the index of the currently selected cr
        index_cr = current_indices.index(selected_cr)

        # make this trace lightgrey
        interactive_layout.update_traces(
            line=transparent_line,
            selector={"meta": selected_cr},
        )

        # in/decrement the cr
        selected_cr = display_to_real[real_to_display[selected_cr] - 1 if left else real_to_display[selected_cr] + 1]

        # and set the selected_cr on the index
        current_indices[index_cr] = selected_cr

        # update this trace to turn black
        interactive_layout.update_traces(
            line=solid_line,
            selector={"meta": selected_cr},
        )

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
        )

    def move_left_callback(button):
        set_current_cr(left=True)

    def move_right_callback(button):
        set_current_cr(left=False)

    def reset_button_callback(button):
        """
        Function to reset the currently selected cable roads
        """
        nonlocal current_indices
        nonlocal selected_cr
        nonlocal layout_overview_df

        # reset the selected cr to none
        selected_cr = None

        # reset the current indices
        current_indices = []

        # reset the layout
        update_line_colors_by_indices(
            [],
            interactive_layout,
            color_map,
            forest_area_3,
            model_list,
            hide_unselected=False,
        )

        # reset the tables
        current_cable_roads_table_figure.data[0].cells.values = [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ]

        layout_overview_table_figure.data[0].cells.values = [
            layout_overview_df[col] for col in layout_overview_df.columns
        ]

        highlight_layout_row(None)
        update_selected_marker(None)

    def create_buttons():
        """
        Define the buttons for interacting with the layout and the comparison table
        """
        move_left_button = Button(description="<-")
        move_right_button = Button(description="->")
        reset_all__CRs_button = Button(
            description="Reset all CRs",
            button_style="danger",
            layout=Layout(width="150px", margin="10px 0 0 auto")
        )

        # and bind all the functions to the buttons
        move_left_button.on_click(move_left_callback)
        move_right_button.on_click(move_right_callback)
        reset_all__CRs_button.on_click(reset_button_callback)

        return (
            move_left_button,
            move_right_button,
            reset_all__CRs_button,
        )

    buttons = list(create_buttons())
    print("test")

    return (
        interactive_layout,
        current_cable_roads_table_figure,
        layout_overview_table_figure,
        anchor_table_figure,
        road_anchor_table_figure,
        pareto_frontier,
        buttons[0],
        buttons[1],
        buttons[2],
    )
