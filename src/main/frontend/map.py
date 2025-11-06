from os import truncate
from typing import List, Dict, Any, Optional
import ipywidgets as w
import plotly.express as px
import plotly.graph_objects as go


THEME = {"border-radius": "12px"}


class Map:
    """
    This component receives vd.map from VizData and renders:
      - trees (with precomputed per-tree colors for each optimization)
      - cable corridors as polylines
      - tail anchors and road anchors
      - fixed-frame square layout in a green-ish card

    Coloring logic:
    - No optimization selected       -> all trees green            (tree_color_default)
    - An optimization row selected   -> tree_colors_by_model[row_idx]
    - A custom selection of lines    -> tree_colors_by_selection[tuple(lines)]
    - Union selection (all lines)    -> tree_colors_by_union
    """

    def __init__(self, map_data: Dict[str, Any], title: str):
        self.data = map_data

        # neutral colors for "unselected" corridors and markers
        self._neutral_line = "rgba(120,120,120,0.45)"
        self._neutral_marker = "rgba(120,120,120,0.65)"

        # base palette for selected corridors
        self._palette = px.colors.qualitative.Plotly

        # lookup real corridor index -> filtered display id
        dl = map_data.get("display_lookup", {})
        self._display_lookup = {int(k): int(v) for k, v in dl.items()}

        # keep track of which corridors are active
        self._selected_set: set[int] = set()

        # (optional) legend widgets
        self._legend_items: Dict[int, Dict[str, Any]] = {}

        # build the figure once
        self.fig = self._build_base_figure()

        # make sure axes show the bbox (from data_prep) with no extra huge padding
        self._apply_axis_ranges(pad_ratio=0.35)

        # pretty title widget (outside the figure)
        self._title_html = w.HTML(
                (
                    "<div style='"
                    "font-weight:800;"
                    "text-align:left;"
                    "margin:0 0 6px 0;"
                    "width:100%;"
                    "font-size:18px;"
                    "'>"
                    f"{title}"
                    "</div>"
                ),
                layout=w.Layout(width="auto")
            )

        # CSS helper to round the border box
        self._BORDER_RADIUS_CSS = w.HTML(
            "<style>.border-radius, .border-radius { "
            f"border-radius: {THEME['border-radius']} !important; "
            "}</style>",
            layout=w.Layout(display="none"),
        )

        # fixed-size card (so border lines up and doesn't stretch weirdly)
        fixed_w, fixed_h = 1200, 900
        self.fig_card = w.Box(
            [self.fig],
            layout=w.Layout(
                background_color="rgb(241, 248, 241)",
                border="2px solid #94b48a",
                overflow="hidden",
                width=f"{fixed_w}px",
                height=f"{fixed_h}px",
                min_width=f"{fixed_w}px",
                max_height=f"{fixed_h}px",
                flex="0 0 auto",
            ),
        )
        self.fig_card.add_class("border-radius")

        self._scroll_wrapper = w.Box(
            [self.fig_card],
            layout=w.Layout(
                overflow_x="scroll",
                overflow_y="hidden",
                width="100%",
                max_width="100%",
                min_width="0",
            )
        )

        # vertical stack = title + card
        self._stack = w.VBox(
            [self._title_html, self._scroll_wrapper],
            layout=w.Layout(width="100%"),
        )

        # outer container that interface.py can place directly
        self.container = w.HBox(
            [self._BORDER_RADIUS_CSS, self._stack],
            layout=w.Layout(align_items="flex-start", gap="16px"),
        )

    # ------------------------------------------------------------------
    # Public API for interface.py
    # ------------------------------------------------------------------

    def get_map_widget(self) -> w.Widget:
        """Return the widget you can drop into your interface layout."""
        return self.container

    def update_map(
        self,
        selected_index: Optional[int] = None,
        selected_lines: Optional[List[int]] = None,
    ) -> None:
        """
        Recolor and toggle visibility:
        - corridors:
            * none selected -> all corridors visible in neutral gray
            * selection -> only those corridors visible in palette colors
        - trees:
            * choose correct precomputed color list from self.data
        """

        # Normalize "selected_lines" into a clean Python list[int]
        ordered_indices: List[int] = []
        if selected_lines is not None:
            if hasattr(selected_lines, "tolist"):
                selected_lines = selected_lines.tolist()
            if isinstance(selected_lines, (list, tuple)):
                ordered_indices = [int(x) for x in selected_lines]
            elif selected_lines != []:
                ordered_indices = [int(selected_lines)]

        # Keep the tree layers visible always
        for tr in self.fig.data:
            if getattr(tr, "meta", None) == "legend-only":
                tr.visible = True
                continue
            if getattr(tr, "name", None) == "trees":
                tr.visible = True

        # ------------------------------------------------------------------
        # CASE A: no optimization row / no corridors selected
        # -> all corridors neutral gray, trees solid green
        # ------------------------------------------------------------------
        if len(ordered_indices) == 0:
            self._selected_set = set()

            for tr in self.fig.data:
                if getattr(tr, "meta", None) == "legend-only":
                    tr.visible = True
                    continue
                lg = getattr(tr, "legendgroup", None)
                if lg == "line":
                    tr.visible = True
                    tr.line.color = self._neutral_line
                    tr.line.width = 0.8
                elif lg in ("tail-conn", "road-conn"):
                    tr.visible = True
                    tr.line.color = self._neutral_line
                    tr.line.width = 1.2
                elif lg in ("tail-marker", "road-marker", "support-marker", "endmast-marker"):
                    tr.visible = True
                    tr.marker.color = self._neutral_marker

            # trees = precomputed default colors (all green)
            default_colors = self.data.get("tree_color_default", None)
            if default_colors:
                self.fig.data[0].marker.color = default_colors
            else:
                self.fig.data[0].marker.color = "green"

            # lock axes to bbox again
            self._apply_axis_ranges(pad_ratio=0.35)

            try:
                self.fig.batch_animate()
            except Exception:
                pass
            return

        # ------------------------------------------------------------------
        # CASE B: we have a corridor selection
        # -> hide all corridors first;
        # -> bring back only selected corridors in vivid colors
        # ------------------------------------------------------------------
        for tr in self.fig.data:
            meta = getattr(tr, "meta", None)
            if meta in ("legend-only", "trees"):
                continue
            tr.visible = False

        # stable order without duplicates
        sel_order = list(dict.fromkeys(int(x) for x in ordered_indices))
        self._selected_set = set(sel_order)

        # color mapping for corridors
        sel_color_map = {
            real_idx: self._palette[i % len(self._palette)]
            for i, real_idx in enumerate(sel_order)
        }

        # Turn on each chosen corridor + anchors with its color
        for real_idx in sel_order:
            c = sel_color_map[int(real_idx)]
            for tr in self.fig.data:
                if getattr(tr, "meta", None) != int(real_idx):
                    continue
                lg = getattr(tr, "legendgroup", None)

                if lg == "line":
                    tr.visible = True
                    tr.line.color = c
                    tr.line.width = 4.5

                elif lg in ("tail-marker", "road-marker", "endmast-marker"):
                    tr.visible = True
                    tr.marker.color = c

                elif lg in ("tail-conn", "road-conn"):
                    tr.visible = True
                    tr.line.color = c
                    tr.line.width = 1.6

                elif lg == "support-marker":
                    tr.visible = True
                    tr.marker.color = c

        # ------------------------------------------------------------------
        # TREE COLORS for the selection
        # Priority:
        #   1. selected_index (model row index)
        #   2. tuple(ordered_indices) from selected_lines
        #   3. union colors if this matches union
        #   4. fallback: default all-green
        # ------------------------------------------------------------------
        colors_for_trees = None

        # 1) if interface gave us the model row index
        if selected_index is not None:
            model_tree_colors = self.data.get("tree_colors_by_model", {})
            colors_for_trees = model_tree_colors.get(int(selected_index))

        # 2) if only lines are passed (maybe manual selection)
        if colors_for_trees is None and ordered_indices:
            key = tuple(ordered_indices)
            by_sel = self.data.get("tree_colors_by_selection", {})
            colors_for_trees = by_sel.get(key)

        # 3) if they're effectively "union"
        if colors_for_trees is None:
            indices_to_show = self.data.get("indices_to_show", [])
            if indices_to_show and tuple(ordered_indices) == tuple(indices_to_show):
                colors_for_trees = self.data.get("tree_colors_by_union")

        # 4) fallback to default (all green)
        if not colors_for_trees:
            colors_for_trees = self.data.get(
                "tree_color_default",
                ["green"] * len(self.fig.data[0].x),
            )

        # actually apply new marker colors to tree layer
        self.fig.data[0].marker.color = colors_for_trees

        # keep bbox (no extra padding because we already padded 10m in data_prep)
        self._apply_axis_ranges(pad_ratio=0.35)

        try:
            self.fig.batch_animate()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_base_figure(self) -> go.FigureWidget:
        """
        Create the static traces:
          - all trees
          - (optional) support trees
          - each corridor polyline
          - each corridor's tail & road anchors + connectors
        """
        fig = go.FigureWidget()

        # ----------------------------------------------------------
        # 1. Trees layer
        # ----------------------------------------------------------
        tree_x = self.data.get("tree_x", [])
        tree_y = self.data.get("tree_y", [])
        tree_bhd = self.data.get("tree_bhd_cm", [])

        tree_custom = [[b] for b in tree_bhd] if len(tree_bhd) == len(tree_x) else None

        default_tree_color = self.data.get("tree_color_default", "green")

        fig.add_trace(
            go.Scatter(
                x=tree_x,
                y=tree_y,
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=5,
                    color=default_tree_color,
                ),
                name="Baum",
                legendrank=0,
                customdata=tree_custom,
                hovertemplate=(
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<br>"
                    "BHD: %{customdata[0]:.1f} cm<extra></extra>"
                )
                if tree_custom
                else "X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                showlegend=False,
                legendgroup="trees",
                meta="trees",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=6,
                    color=self._neutral_marker,
                ),
                name="Bäume",
                legendrank=0,
                showlegend=True,
                legendgroup="trees",
                hoverinfo="skip",
                meta="legend-only",
            )
        )

        # ----------------------------------------------------------
        # 2. Optional support/mast trees
        # ----------------------------------------------------------
        support_by_corr = self.data.get("support_trees_by_corridor", {})
        support_present = False

        if support_by_corr:
            for real_idx, tree_indices in support_by_corr.items():
                sup_x = [tree_x[i] for i in tree_indices if 0 <= i < len(tree_x)]
                sup_y = [tree_y[i] for i in tree_indices if 0 <= i < len(tree_x)]
                if not sup_x:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=sup_x,
                        y=sup_y,
                        mode="markers",
                        marker=dict(
                            symbol="square",
                            size=8,
                            color=self._neutral_marker,
                            line=dict(width=0),
                        ),
                        name="Stützbaum",
                        meta=int(real_idx),
                        legendgroup="support-marker",
                        legendrank=20,
                        hovertemplate="Stützmast<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                        showlegend=False,
                    )
                )
                support_present = True

        if not support_present:
            support_mask = self.data.get("support_tree_mask", None)
            if isinstance(support_mask, (list, tuple)) and len(support_mask) == len(tree_x):
                sup_x = [tree_x[i] for i, m in enumerate(support_mask) if m]
                sup_y = [tree_y[i] for i, m in enumerate(support_mask) if m]
            else:
                sup_x, sup_y = [], []

            if sup_x:
                fig.add_trace(
                    go.Scatter(
                        x=sup_x,
                        y=sup_y,
                        mode="markers",
                        marker=dict(
                            symbol="square",
                            size=8,
                            color=self._neutral_marker,
                            line=dict(width=0),
                        ),
                        name="Stützbäume",
                        legendgroup="support-marker",
                        legendrank=20,
                        hovertemplate="Stützmast<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                        showlegend=False,
                    )
                )
                support_present = True

        if support_present:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=8,
                        color=self._neutral_marker,
                        line=dict(width=0),
                    ),
                    name="Stützbaum",
                    legendgroup="legend-support",
                    legendrank=20,
                    hoverinfo="skip",
                    showlegend=True,
                    meta="legend-only",
                )
            )

        # ----------------------------------------------------------
        # 3. Corridors + anchors
        # ----------------------------------------------------------
        corridors = self.data.get("corridors", {})
        tail_present = False
        road_present = False
        endmast_present = False
        for real_idx, corr in corridors.items():
            xs = corr.get("xs", [])
            ys = corr.get("ys", [])
            display_id = self._display_lookup.get(int(real_idx), corr.get("display_id", int(real_idx)))

            # Pre-store hover data (corridor id, length, volume)
            line_len = corr.get("length_m", 0.0)
            line_vol = corr.get("volume_m3", 0.0)
            base_custom = [int(display_id), float(line_len), float(line_vol)]
            cd = [base_custom[:] for _ in xs]

            # main cable corridor polyline, initially neutral gray
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=self._neutral_line, width=0.8),
                    name=f"{int(display_id)}",
                    meta=int(real_idx),
                    legendgroup="line",
                    legendrank=50,
                    customdata=cd,
                    hovertemplate=(
                        "Seiltrasse: %{customdata[0]}<br>"
                        "Seillänge: %{customdata[1]:.1f} m<br>"
                        "Volumen: %{customdata[2]:.1f} m³<extra></extra>"
                    ),
                    visible=True,
                )
            )

            # Tail anchor marker
            tail = corr.get("tail_anchor", {})
            t_cd = [[tail.get("BHD")]] if tail.get("BHD") is not None else None

            if tail.get("x") is not None and tail.get("y") is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[tail.get("x")],
                        y=[tail.get("y")],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=11,
                            color=self._neutral_marker,
                        ),
                        name="Tal Ankerbaum",
                        meta=int(real_idx),
                        legendgroup="tail-marker",
                        legendrank=10,
                        hovertemplate=(
                            "Ankerbaum<br>"
                            "X: %{x:.2f}<br>"
                            "Y: %{y:.2f}"
                            + (
                                "<br>BHD: %{customdata[0]:.1f} cm"
                                if t_cd
                                else ""
                            )
                            + "<extra></extra>"
                        ),
                        customdata=t_cd,
                        visible=True,
                        showlegend=False,
                    )
                )
                tail_present = True

            # Connector from corridor end -> tail anchor
            if tail.get("x") is not None and tail.get("y") is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[corr["end"][0], tail.get("x")],
                        y=[corr["end"][1], tail.get("y")],
                        mode="lines",
                        line=dict(
                            dash="dot",
                            width=1.2,
                            color=self._neutral_line,
                        ),
                        name="",
                        meta=int(real_idx),
                        legendgroup="tail-conn",
                        hoverinfo="skip",
                        visible=True,
                        showlegend=False,
                    )
                )

            # Endmast marker (prefer end support tree location)
            endmast = corr.get("endmast", {})
            em_x = endmast.get("x")
            em_y = endmast.get("y")
            if em_x is not None and em_y is not None:
                em_hover = "Endmast<br>X: %{x:.2f}<br>Y: %{y:.2f}"
                em_custom = []
                if endmast.get("BHD") is not None:
                    em_hover += "<br>BHD: %{customdata[0]:.1f} cm"
                    em_custom.append(float(endmast["BHD"]))
                if endmast.get("h") is not None:
                    next_idx = len(em_custom)
                    em_hover += f"<br>Höhe: %{{customdata[{next_idx}]:.1f}} m"
                    em_custom.append(float(endmast["h"]))

                fig.add_trace(
                    go.Scatter(
                        x=[em_x],
                        y=[em_y],
                        mode="markers",
                        marker=dict(
                            symbol="diamond",  # distinct mast marker
                            size=10,
                            color=self._neutral_marker,
                        ),
                        name="Endmast",
                        meta=int(real_idx),
                        legendgroup="endmast-marker",
                        legendrank=15,
                        hovertemplate=em_hover + "<extra></extra>",
                        customdata=[em_custom] if em_custom else None,
                        showlegend=False,
                        visible=True,
                    )
                )
                endmast_present = True

            # Road anchors (possibly multiple), plus dotted connectors from start
            for ra in corr.get("road_anchors", []):
                ra_cd = [[ra.get("BHD")]] if ra.get("BHD") is not None else None

                # Road anchor marker
                fig.add_trace(
                    go.Scatter(
                        x=[ra.get("x")],
                        y=[ra.get("y")],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color=self._neutral_marker,
                        ),
                        name="Straßen Ankerbaum",
                        meta=int(real_idx),
                        legendgroup="road-marker",
                        legendrank=30,
                        hovertemplate=(
                            "Ankerbaum<br>"
                            "X: %{x:.2f}<br>"
                            "Y: %{y:.2f}"
                            + (
                                "<br>BHD: %{customdata[0]:.1f} cm"
                                if ra_cd
                                else ""
                            )
                            + "<extra></extra>"
                        ),
                        customdata=ra_cd,
                        visible=True,
                        showlegend=False,
                    )
                )

                road_present = True

                # dotted connector start -> road anchor
                fig.add_trace(
                    go.Scatter(
                        x=[corr["start"][0], ra.get("x")],
                        y=[corr["start"][1], ra.get("y")],
                        mode="lines",
                        line=dict(
                            dash="dot",
                            width=1.2,
                            color=self._neutral_line,
                        ),
                        name="",
                        meta=int(real_idx),
                        legendgroup="road-conn",
                        hoverinfo="skip",
                        visible=True,
                        showlegend=False,
                    )
                )
        if tail_present:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=11,
                        color=self._neutral_marker,
                    ),
                    name="Tal Ankerbaum",
                    legendgroup="tail-marker",
                    legendrank=10,
                    hoverinfo="skip",
                    showlegend=True,
                    meta="legend-only",
                )
            )

        if road_present:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color=self._neutral_marker,
                    ),
                    name="Straßen Ankerbaum",
                    legendgroup="road-marker",
                    legendrank=30,
                    hoverinfo="skip",
                    showlegend=True,
                    meta="legend-only",
                )
            )

        if endmast_present:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=10,
                        color=self._neutral_marker,
                    ),
                    name="Endmast",
                    legendgroup="endmast-marker",
                    legendrank=15,
                    hoverinfo="skip",
                    showlegend=True,
                    meta="legend-only",
                )
            )

        # ----------------------------------------------------------
        # 4. Layout and axes styling
        # ----------------------------------------------------------
        fixed_w, fixed_h = 1200, 900
        fig.update_layout(
            autosize=False,
            width=fixed_w,
            height=fixed_h,
            margin=dict(r=20, l=20, t=30, b=20),
            paper_bgcolor="rgb(241, 248, 241)",
            plot_bgcolor="rgb(241, 248, 241)",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            legend=dict(itemsizing="constant"),
        )

        # Lock meters: 1:1 aspect ratio (no stretching)
        fig.update_xaxes(scaleanchor="y", scaleratio=0.5)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # apply VizData-provided extents once
        if self.data.get("x_range"):
            fig.update_xaxes(range=list(self.data["x_range"]))
        if self.data.get("y_range"):
            fig.update_yaxes(range=list(self.data["y_range"]))

        return fig

    def _apply_axis_ranges(self, pad_ratio: float = 0.0) -> None:
        """
        Force a square-ish viewport based on the global bounding box in map_data.
        We already padded the bbox by 10m in data_prep, so default pad_ratio=0.0.
        """
        xr = self.data.get("x_range")
        yr = self.data.get("y_range")
        if xr is None or yr is None:
            return

        minx, maxx = xr
        miny, maxy = yr

        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)

        span = max(maxx - minx, maxy - miny)
        span *= (1.0 + pad_ratio)
        half = 0.5 * span

        target_x = [cx - half, cx + half]
        target_y = [cy - half, cy + half]

        self.fig.update_xaxes(range=target_x, scaleanchor="y", scaleratio=1)
        self.fig.update_yaxes(range=target_y, scaleanchor="x", scaleratio=1)
