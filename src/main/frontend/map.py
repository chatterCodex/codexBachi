from typing import List, Dict, Any, Optional
import math
import ipywidgets as w
import plotly.express as px
import plotly.graph_objects as go


THEME = {
    "border-radius": "12px",
}


class Map:
    def __init__(self, map_data: Dict[str, Any], title: str):
        """
        map_data is vd.map from VizData, e.g.:

        {
            "tree_x": [...],
            "tree_y": [...],
            "tree_bhd_cm": [...],
            "tree_assignment": [...],            # per-tree real corridor index
            "color_map": { real_idx: "#hex", ...},
            "corridors": {
                real_idx: {
                    "xs": [...],
                    "ys": [...],
                    "start": (x0,y0),
                    "end":   (x1,y1),
                    "tail_anchor": {"x":..,"y":..,"BHD":..},
                    "road_anchors": [{"x":..,"y":..,"BHD":..}, ...],
                    "length_m": float,
                    "volume_m3": float,
                },
                ...
            },
            "x_range": (minx, maxx),
            "y_range": (miny, maxy),
        }
        """
        self.data = map_data

        # soft neutral styles (no harsh black)
        self._neutral_line = "rgba(120,120,120,0.45)"
        self._neutral_marker = "rgba(120,120,120,0.65)"

        # plotly qualitative palette (used for selected corridors)
        self._palette = px.colors.qualitative.Plotly

        # which real corridor indices are currently selected/highlighted
        self._selected_set: set[int] = set()

        # (optional) external legend widgets, if you ever add them
        self._legend_items: Dict[int, Dict[str, Any]] = {}

        # build base figure
        self.fig = self._build_base_figure()

        # make sure the axes are using a global square view (not zoomed to first traces)
        self._apply_axis_ranges(pad_ratio=0.2)

        # title above plot
        self._title_html = w.HTML(
            f"<div style='font-weight:600; font-size:16px; line-height:1.2; margin:0 0 8px 0;'>{title}</div>",
        )

        # CSS helper for rounded border
        self._BORDER_RADIUS_CSS = w.HTML(
            (
                "<style>"
                ".border-radius, .border-radius { "
                f"border-radius: {THEME['border-radius']} !important;"
                "}"
                "</style>"
            ),
            layout=w.Layout(display="none"),
        )

        # fixed-size card around the figure so the green frame lines up perfectly
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
            ),
        )
        self.fig_card.add_class("border-radius")

        # vertical stack: title + figure card
        self._stack = w.VBox(
            [self._title_html, self.fig_card],
            layout=w.Layout(width="auto"),
        )

        # outer container shown in the UI
        self.container = w.HBox(
            [self._BORDER_RADIUS_CSS, self._stack],
            layout=w.Layout(align_items="flex-start", gap="16px"),
        )

    # ---------------- Public API ----------------

    def get_map_widget(self) -> w.Widget:
        """Return the widget to be inserted into the main interface layout."""
        return self.container

    def update_map(
        self,
        selected_index: Optional[int] = None,
        selected_lines: Optional[List[int]] = None,
    ) -> None:
        """
        Called from interface._on_select_with_vd().
        - selected_lines is a list of REAL corridor indices of the chosen optimization result.
        - If nothing is selected, show ALL corridors in neutral gray,
          and paint ALL trees green.
        - If something is selected, only those corridors get vivid colors,
          and trees are colored by their assigned corridor if that corridor is selected.
        """
        # normalize indices
        indices: List[int] = []
        if selected_lines is not None:
            if hasattr(selected_lines, "tolist"):
                selected_lines = selected_lines.tolist()
            if isinstance(selected_lines, (list, tuple)):
                indices = [int(x) for x in selected_lines]
            elif selected_lines != []:
                indices = [int(selected_lines)]

        # keep tree layers visible always; we recolor markers below
        for tr in self.fig.data:
            if getattr(tr, "name", None) in ("Bäume", "Stützbäume"):
                tr.visible = True

        # ------------------------------------------------------------------
        # CASE 1: no optimization result selected
        # ------------------------------------------------------------------
        if len(indices) == 0:
            self._selected_set = set()

            # show every corridor in neutral gray
            for tr in self.fig.data:
                lg = getattr(tr, "legendgroup", None)
                if lg == "line":
                    # cable corridor polyline
                    tr.visible = True
                    tr.line.color = self._neutral_line
                    tr.line.width = 0.8
                    # neutral hover is fine; leave template as-is
                elif lg in ("tail-conn", "road-conn"):
                    # dotted helper connections
                    tr.visible = True
                    tr.line.color = self._neutral_line
                    tr.line.width = 1.2
                elif lg in ("tail-marker", "road-marker"):
                    # triangle markers at anchors
                    tr.visible = True
                    tr.marker.color = self._neutral_marker

            # all trees pure green in this state
            if len(self.fig.data) > 0:
                self.fig.data[0].marker.color = "green"

            # legend maintenance if you add external legend later
            for ridx, item in self._legend_items.items():
                item["chk"].value = False
                item["chk"].disabled = False
                self._paint_swatch(item["box"], self._neutral_line)

            # keep global extents (don't zoom in on anything)
            self._apply_axis_ranges(pad_ratio=0.2)

            try:
                self.fig.batch_animate()
            except Exception:
                pass
            return

        # ------------------------------------------------------------------
        # CASE 2: we have a selection (an optimization result is chosen)
        # ------------------------------------------------------------------
        # hide everything by default (except trees)
        for tr in self.fig.data:
            if getattr(tr, "name", None) in ("Bäume", "Stützbäume"):
                continue
            tr.visible = False

        # consistent color mapping for these selected corridors
        # order is stable, first occurrence wins
        sel_order = list(dict.fromkeys(int(x) for x in indices))
        sel_color_map = {
            real_idx: self._palette[i % len(self._palette)]
            for i, real_idx in enumerate(sel_order)
        }
        self._selected_set = set(sel_order)

        # activate and color only the selected corridors / anchors
        for real_idx in sel_order:
            cr_color = sel_color_map[int(real_idx)]

            for tr in self.fig.data:
                if getattr(tr, "meta", None) != int(real_idx):
                    continue

                lg = getattr(tr, "legendgroup", None)

                if lg == "line":
                    tr.visible = True
                    tr.line.color = cr_color
                    tr.line.width = 4.5
                    # vivid hover with length + volume (customdata was attached)
                    tr.hovertemplate = (
                        "Seiltrasse %{customdata[0]}<br>"
                        "Seillänge: %{customdata[1]:.1f} m<br>"
                        "Volumen: %{customdata[2]:.1f} m³<extra></extra>"
                    )

                elif lg in ("tail-marker", "road-marker"):
                    tr.visible = True
                    tr.marker.color = cr_color

                elif lg in ("tail-conn", "road-conn"):
                    tr.visible = True
                    tr.line.color = cr_color
                    tr.line.width = 1.6

        # update external legend swatches (if any UI around it exists)
        for ridx, item in self._legend_items.items():
            if ridx in self._selected_set:
                item["chk"].value = True
                item["chk"].disabled = False
                self._paint_swatch(item["box"], sel_color_map[ridx])
            else:
                item["chk"].value = False
                item["chk"].disabled = False
                self._paint_swatch(item["box"], self._neutral_line)

        # recolor trees:
        #   - trees assigned to selected corridors get that corridor's color
        #   - all other trees are dimmed gray
        self._apply_tree_colors_selected(sel_order, sel_color_map)

        # lock back to global view so we don't zoom to only chosen lines
        self._apply_axis_ranges(pad_ratio=0.2)

        try:
            self.fig.batch_animate()
        except Exception:
            pass

    # ---------------- Internals ----------------

    def _build_base_figure(self) -> go.FigureWidget:
        """
        Build the static layers:
        - tree scatter
        - (optional) support trees
        - cable corridors (polylines)
        - anchors and connector lines
        """
        fig = go.FigureWidget()

        # --- Tree layer ----------------------------------------------------
        tree_x = self.data.get("tree_x", [])
        tree_y = self.data.get("tree_y", [])
        tree_bhd = self.data.get("tree_bhd_cm", [])

        # attach BHD per tree in hover
        if len(tree_bhd) == len(tree_x):
            tree_custom = [[b] for b in tree_bhd]
        else:
            tree_custom = None

        fig.add_trace(
            go.Scatter(
                x=tree_x,
                y=tree_y,
                mode="markers",
                marker=dict(symbol="circle", size=5, color="green"),  # default state: all green
                name="Bäume",
                hovertemplate=(
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<br>"
                    "BHD: %{customdata[0]:.1f} cm<extra></extra>"
                )
                if tree_custom
                else (
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<extra></extra>"
                ),
                customdata=tree_custom,
                showlegend=False,
            )
        )

        # --- Support / mast trees (optional square markers) ----------------
        # If mask provided (bool per tree), highlight those trees as squares.
        support_mask = self.data.get("support_tree_mask", None)
        if isinstance(support_mask, (list, tuple)) and len(support_mask) == len(tree_x):
            sup_x = [tree_x[i] for i, m in enumerate(support_mask) if m]
            sup_y = [tree_y[i] for i, m in enumerate(support_mask) if m]
        else:
            sup_x, sup_y = [], []

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
                hovertemplate="Stützmast<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )

        # --- Cable corridors + anchors -------------------------------------
        corridors_data: Dict[Any, Any] = self.data.get("corridors", {})
        for real_idx, corr in corridors_data.items():
            xs = corr.get("xs", [])
            ys = corr.get("ys", [])

            # length & volume (from precomputed map payload)
            line_len = corr.get("length_m", 0.0)
            line_vol = corr.get("volume_m3", 0.0)

            # we want stable per-line hover even when recoloring later
            cd = [[int(real_idx), float(line_len), float(line_vol)]]

            # main corridor polyline
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=self._neutral_line, width=0.8),
                    name=f"{int(real_idx) + 1}",
                    meta=int(real_idx),
                    legendgroup="line",
                    hovertemplate=(
                        "Seiltrasse %{customdata[0]}<br>"
                        "Seillänge: %{customdata[1]:.1f} m<br>"
                        "Volumen: %{customdata[2]:.1f} m³<extra></extra>"
                    ),
                    customdata=cd,
                    visible=True,
                )
            )

            # tail anchor marker
            tail = corr.get("tail_anchor", {})
            t_bhd = tail.get("BHD", None)
            if t_bhd is not None:
                t_cd = [[t_bhd]]
            else:
                t_cd = None

            fig.add_trace(
                go.Scatter(
                    x=[tail.get("x")],
                    y=[tail.get("y")],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=11,
                        color=self._neutral_marker,
                    ),
                    name="Tail Anchor",
                    meta=int(real_idx),
                    legendgroup="tail-marker",
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

            # connector line: corridor end -> tail anchor
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

            # road anchors and connectors from start
            for ra in corr.get("road_anchors", []):
                ra_bhd = ra.get("BHD", None)
                if ra_bhd is not None:
                    ra_cd = [[ra_bhd]]
                else:
                    ra_cd = None

                # road anchor marker
                fig.add_trace(
                    go.Scatter(
                        x=[ra.get("x")],
                        y=[ra.get("y")],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color=self._neutral_marker,
                        ),
                        name="Road Anchor",
                        meta=int(real_idx),
                        legendgroup="road-marker",
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

                # dotted connector from corridor start to road anchor
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

        # --- Figure layout -------------------------------------------------
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

        # lock aspect ratio so x/y are same scale (meters → meters)
        fig.update_xaxes(scaleanchor="y", scaleratio=1)

        # if VizData already gave us ranges, apply them tentatively
        if self.data.get("x_range"):
            fig.update_xaxes(range=list(self.data["x_range"]))
        if self.data.get("y_range"):
            fig.update_yaxes(range=list(self.data["y_range"]))

        return fig

    # ---- Coloring helpers for trees --------------------------------------

    def _apply_tree_colors_selected(
        self,
        sel_order: List[int],
        sel_color_map: Dict[int, str],
    ) -> None:
        """
        When corridors are selected:
        - trees assigned to selected corridors get that corridor's color
        - all other trees go neutral gray
        """
        if not self.fig.data:
            return
        if self.fig.data[0].name != "Bäume":
            return

        tree_assign = self.data.get("tree_assignment", None)
        if not (
            isinstance(tree_assign, (list, tuple))
            and len(self.fig.data[0].x) == len(tree_assign)
        ):
            # fallback: just leave trees green
            self.fig.data[0].marker.color = "green"
            return

        sel_set = set(sel_order)

        def _sel_color(lbl):
            try:
                ilbl = int(lbl)
            except Exception:
                return self._neutral_marker
            # dim invalid/negative
            if ilbl < 0:
                return self._neutral_marker
            # highlight only selected corridors
            if ilbl in sel_set:
                return sel_color_map.get(
                    ilbl,
                    self._palette[ilbl % len(self._palette)],
                )
            return self._neutral_marker

        self.fig.data[0].marker.color = [_sel_color(lbl) for lbl in tree_assign]

    # ---- Axis helper -----------------------------------------------------

    def _apply_axis_ranges(self, pad_ratio: float = 0.2) -> None:
        """
        Force a global square viewport that covers all trees/corridors.
        pad_ratio is extra fractional padding around the bounding box.
        This keeps the map from zooming in too much and keeps the green
        border box nicely aligned.
        """
        # prefer the precomputed extents from VizData
        xr = self.data.get("x_range")
        yr = self.data.get("y_range")

        if xr is None or yr is None:
            # fallback: compute from traces
            all_x: List[float] = []
            all_y: List[float] = []
            for tr in self.fig.data:
                xs = getattr(tr, "x", None)
                ys = getattr(tr, "y", None)
                if xs is not None:
                    all_x.extend(v for v in xs if v is not None)
                if ys is not None:
                    all_y.extend(v for v in ys if v is not None)

            if not all_x or not all_y:
                return

            minx, maxx = min(all_x), max(all_x)
            miny, maxy = min(all_y), max(all_y)
        else:
            minx, maxx = xr
            miny, maxy = yr

        # center + span
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        span_x = maxx - minx
        span_y = maxy - miny
        span = max(span_x, span_y)

        # pad and make square
        span *= (1.0 + pad_ratio)
        half = 0.5 * span

        final_x = (cx - half, cx + half)
        final_y = (cy - half, cy + half)

        self.fig.update_xaxes(range=list(final_x), scaleanchor="y", scaleratio=1)
        self.fig.update_yaxes(range=list(final_y))

    # ---- Misc small helpers ---------------------------------------------

    def _polyline_length(self, xs: List[float], ys: List[float]) -> float:
        """Euclidean length of a polyline given x/y vertices."""
        if not xs or not ys or len(xs) != len(ys):
            return 0.0
        total = 0.0
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            total += math.hypot(dx, dy)
        return total

    def _paint_swatch(self, box: w.HTML, color: str) -> None:
        """If you ever wire up external legend items, this colors their little squares."""
        box.value = (
            "<div "
            "style='width:14px;height:14px;"
            f"background:{color};border:1px solid #777;'>"
            "</div>"
        )
