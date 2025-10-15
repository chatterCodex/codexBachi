from typing import List, Dict, Any, Optional
import ipywidgets as w
import plotly.express as px
import plotly.graph_objects as go
from scipy.__config__ import show

THEME = {
    "border-radius": "12px",
}

class Map:
    def __init__(self, map_data: Dict[str, Any]):
        self.data = map_data

        # define everything used by _build_base_figure / legend BEFORE building
        self._legend_items: Dict[int, Dict[str, Any]] = {}
        self._neutral_line = "rgba(0,0,0,0.4)"
        self._neutral_marker = "rgba(0,0,0,0.6)"
        self._palette = px.colors.qualitative.Plotly
        self._selected_set: set[int] = set()

        # now we can build the figure safely
        self.fig = self._build_base_figure()

        # then build the side legend (needs _legend_items/_selected_set present)

        self._BORDER_RADIUS_CSS = w.HTML(
            f"<style>.border-radius, .border-radius {{ border-radius: {THEME['border-radius']} !important; }}</style>",
            layout=w.Layout(display="none"),
        )

        # --- wrap the figure in a bordered, rounded, soft-green card ---
        self.fig_card = w.Box(
            [self.fig],
            layout=w.Layout(
                background_color="rgb(241, 248, 241)",
                border="2px solid #94b48a",
            ),
        )

        self.fig_card.add_class("border-radius")

        # container: include the CSS injector, then the card + legend
        self.container = w.HBox(
            [self._BORDER_RADIUS_CSS, self.fig_card],
            layout=w.Layout(align_items="flex-start", gap="16px"),
        )

    # ---------------- Public API ----------------
    def update_map(self, selected_index: Optional[int] = None, selected_lines: Optional[List[int]] = None) -> None:
        # Normalize selected_lines => list[int]
        indices: List[int] = []
        if selected_lines is not None:
            if hasattr(selected_lines, "tolist"):  # numpy/pandas
                selected_lines = selected_lines.tolist()
            if isinstance(selected_lines, (list, tuple)):
                indices = [int(x) for x in selected_lines]
            elif selected_lines != []:
                indices = [int(selected_lines)]

        for tr in self.fig.data:
            if getattr(tr, "name", None) == "Bäume":
                tr.visible = True

        # NO SELECTION branch
        if len(indices) == 0:
            self._selected_set = set()   # <-- NEW
            # ... (leave your neutral recolor code as-is) ...
            for ridx, item in self._legend_items.items():
                item["chk"].value = True
                item["chk"].disabled = False
                self._paint_swatch(item["box"], self._neutral_line)
            # ...
            return

        # WITH SELECTION
        for tr in self.fig.data:
            if getattr(tr, "name", None) == "Bäume":
                continue
            tr.visible = False

        sel_order = list(dict.fromkeys(int(x) for x in indices))
        sel_color_map = {ri: self._palette[i % len(self._palette)] for i, ri in enumerate(sel_order)}

        # <-- NEW: record selection explicitly
        self._selected_set = set(sel_order)

        # show only selected, color them
        for real_idx in sel_order:
            color = sel_color_map[int(real_idx)]
            disp = self.data["real_to_display"].get(int(real_idx), int(real_idx))
            for tr in self.fig.data:
                if getattr(tr, "meta", None) != int(real_idx):
                    continue
                lg = getattr(tr, "legendgroup", None)
                if lg == "line":
                    tr.visible = True
                    tr.line.color = color
                    tr.line.width = 4.5
                    tr.hovertemplate = f"Seiltrasse {disp}<extra></extra>"
                elif lg in ("tail-marker", "road-marker"):
                    tr.visible = True
                    tr.marker.color = color
                elif lg in ("tail-conn", "road-conn"):
                    tr.visible = True
                    tr.line.color = color
                    tr.line.width = 1.6

        # Legend: color selected, grey others; do NOT disable boxes
        for ridx, item in self._legend_items.items():
            if ridx in self._selected_set:
                item["chk"].value = True
                item["chk"].disabled = False
                self._paint_swatch(item["box"], sel_color_map[ridx])
            else:
                # unchecked + grey swatch; user can still re-enable after clearing selection
                item["chk"].value = False
                item["chk"].disabled = False
                self._paint_swatch(item["box"], self._neutral_line)

        try: self.fig.batch_animate()
        except Exception: pass

    def get_map_widget(self) -> w.Widget:
        # Return the HBox (map + legend)
        return self.container

    # ---------------- Internals ----------------
    def _build_base_figure(self) -> go.FigureWidget:
        fig = go.FigureWidget()

        # Trees (always present; we keep them visible explicitly in update_map)
        fig.add_trace(go.Scatter(
            x=self.data["tree_x"], y=self.data["tree_y"],
            mode="markers",
            marker=dict(color="green", size=6),
            name="Bäume",  # sentinel name used in code
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        ))

        # Every corridor: polyline + tail marker/connector + all road markers/connectors
        for real_idx in self.data["indices_to_show"]:
            corr = self.data["corridors"][int(real_idx)]

            # main cable corridor (neutral)
            fig.add_trace(go.Scatter(
                x=corr["xs"], y=corr["ys"],
                mode="lines",
                line=dict(color=self._neutral_line, width=0.8),
                name=f"CR {self.data['real_to_display'].get(int(real_idx), int(real_idx))}",
                meta=int(real_idx),
                legendgroup="line",
                hoverinfo="skip",
            ))

            # tail anchor marker (true anchor coords)
            fig.add_trace(go.Scatter(
                x=[corr["tail_anchor"]["x"]], y=[corr["tail_anchor"]["y"]],
                mode="markers",
                marker=dict(symbol="triangle-up", size=11, color=self._neutral_marker),
                name="Tail Anchor",
                meta=int(real_idx),
                legendgroup="tail-marker",
                hovertemplate="Tail: X %{x:.2f}, Y %{y:.2f}<extra></extra>",
                visible=True,
                showlegend=False,
            ))

            # tail connector (line end → tail anchor)
            fig.add_trace(go.Scatter(
                x=[corr["end"][0], corr["tail_anchor"]["x"]],
                y=[corr["end"][1], corr["tail_anchor"]["y"]],
                mode="lines",
                line=dict(dash="dot", width=1.2, color=self._neutral_line),
                name="",
                meta=int(real_idx),
                legendgroup="tail-conn",
                hoverinfo="skip",
                visible=True,
                showlegend=False,
            ))

            # road anchors + connectors (start → road anchor)
            for ra in corr["road_anchors"]:
                fig.add_trace(go.Scatter(
                    x=[ra["x"]], y=[ra["y"]],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color=self._neutral_marker),
                    name="Road Anchor",
                    meta=int(real_idx),
                    legendgroup="road-marker",
                    hovertemplate="Road: X %{x:.2f}, Y %{y:.2f}<extra></extra>",
                    visible=True,
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=[corr["start"][0], ra["x"]],
                    y=[corr["start"][1], ra["y"]],
                    mode="lines",
                    line=dict(dash="dot", width=1.2, color=self._neutral_line),
                    name="",
                    meta=int(real_idx),
                    legendgroup="road-conn",
                    hoverinfo="skip",
                    visible=True,
                    showlegend=False,
                ))

        # Layout
        fig.update_layout(
            width=1200, height=900,
            margin=dict(r=20, l=20, t=30, b=20),
            paper_bgcolor="rgb(241, 248, 241)",
            plot_bgcolor="rgb(241, 248, 241)",
            xaxis_title="X (m)", yaxis_title="Y (m)",
            legend=dict(itemsizing="constant"),
        )
        if self.data["x_range"]:
            fig.update_xaxes(range=list(self.data["x_range"]))
        if self.data["y_range"]:
            fig.update_yaxes(range=list(self.data["y_range"]))
            
        return fig

    def _is_allowed_by_selection(self, ridx: int) -> bool:
        return (len(self._selected_set) == 0) or (int(ridx) in self._selected_set)

    def _paint_swatch(self, box: w.HTML, color: str) -> None:
        box.value = f"<div style='width:14px;height:14px;background:{color};border:1px solid #777;'></div>"
