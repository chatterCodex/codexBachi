import ipywidgets as w
import ipyevents as ev
import pandas as pd
import plotly.graph_objects as go

import numpy as np
from plotly.colors import hex_to_rgb
from typing import Callable, Dict, List, Optional, Tuple, Sequence, cast

# ---------- Naming & _THEME ----------
_THEME = {
    "card_bg": "rgb(241, 248, 241)",
    "panel_bg": "rgb(255, 255, 255)",
    "grid": "#e0e0e0",
    "card_border": "#94b48a",
    "text": "#0f2010",

    "muted_line": "rgba(120, 120, 120, 0.85)",
    "muted_fill": "rgba(120, 120, 120, 0.14)",
    "muted_text": "#8a8a8a",
    "mini_bg_inactive": "rgba(0, 0, 0, 0.035)",

    "hover_border": "#48723b",

    "card_radius": "12px",
}

# Sort option → key function mapping (descending best)
_SORT_KEYS = {
    "original": None,  # keep initial order
    "ergo": "Ergonomische Optimierung",
    "eco": "Ökologische Optimierung",
    "cost": "Kosten Optimierung",
    "area": "triangle_area",
}
_SORT_LABELS = [
    ("Standard", "original"),
    ("Ergonomische Optimierung", "ergo"),
    ("Ökologische Optimierung", "eco"),
    ("Kosten Optimierung", "cost"),
    ("Gesamtwert", "area"),
]

_CONSTANTS = {
    "sort_bar_height": 32,
    "sort_bar_gap": 10,
}

# ---------- Lightweight CSS helpers ----------

_BORDER_RADIUS_CSS = w.HTML(
    f"<style>.border-radius, .border-radius {{ border-radius: {_THEME['card_radius']} !important; }}</style>",
    layout=w.Layout(display="none"),
)
_POINTER_CSS = w.HTML(
    "<style>.mini-card, .mini-card * { cursor: pointer !important; }</style>",
    layout=w.Layout(display="none"),
)
_DROPDOWN_CSS = w.HTML(
    f"""
<style>
  .app-scope .widget-dropdown select {{
    border-radius: 16px;
    border: 2px solid {_THEME['card_border']};
    background-color: {_THEME['card_bg']};
    color: {_THEME['text']};
    outline: none;
  }}
  .app-scope .widget-dropdown select:hover,
  .app-scope .widget-dropdown select:focus {{
    border-color: {_THEME['hover_border']};
    box-shadow: none;
  }}
  .app-scope .widget-dropdown,
  .app-scope .widget-dropdown *,
  .app-scope .sort-label {{
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Helvetica,Arial,sans-serif;
    font-size: 14px;
  }}
  .app-scope .widget-label {{ background: transparent; }}
</style>
""",
    layout=w.Layout(display="none"),
)


# ---------- Data prep ----------

def _as_float(v: object) -> float:
    """Robust scalar→float that satisfies Pylance and avoids pandas/numpy constructors."""

    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, (bool, np.bool_)):
        return float(v)

    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode()
        except Exception:
            return float("-inf")

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return float("-inf")
        try:
            return float(s)
        except ValueError:
            try:
                return float(s.replace(",", "."))
            except Exception:
                return float("-inf")

    if isinstance(v, np.generic):
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return float("-inf")

    return float("-inf")


# ---------- Plot builders ----------

def _build_big_radar(scores: pd.DataFrame, active_indices: List[int], height: int, width: int, axes: List[str]) -> Tuple[w.VBox, go.FigureWidget, Dict[int, int]]:
    thetas = list(axes) + [axes[0]]

    ticktext = [lbl.replace(" ", "<br>") for lbl in axes]

    marginlr = max(12, int(int(width)*0.08))
    margint = max(12, int(int(height)*0.12))
    marginb = max(8, int(int(height)*0.06))

    fig = go.FigureWidget()
    fig._config = {'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False}
    index_to_trace: Dict[int, int] = {}

    for t_idx, idx in enumerate(scores.index):
        row = scores.loc[idx]
        rs = [_as_float(scores.at[idx, a]) for a in axes]
        r_closed = rs + [rs[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_closed, theta=thetas, name=row["Name"], mode="lines",
            line=dict(color=scores["color"][idx], width=2),
            fill="toself", fillcolor=scores["fill_color"][idx],
            visible=True if idx in active_indices else "legendonly",
        ))
        index_to_trace[idx] = t_idx

    fig.update_layout(
        uirevision="keep",
        paper_bgcolor=_THEME["card_bg"],
        dragmode=False, showlegend=False,
        hoverlabel=dict(bgcolor=_THEME["panel_bg"], font=dict(color=_THEME["text"])),
        polar=dict(
            bgcolor=_THEME["card_bg"],
            radialaxis=dict(range=[0, 1], gridcolor=_THEME["grid"], showticklabels=False, showline=False),
            angularaxis=dict(
                tickmode="array",
                tickvals=axes,
                ticktext=ticktext,
                gridcolor=_THEME["grid"], linecolor=_THEME["grid"], tickfont=dict(color=_THEME["text"]), rotation=90, layer="below traces", showgrid=False),
            gridshape='circular',
        ),
        margin=dict(l=marginlr, r=marginlr, t=margint, b=marginb),
        width=width,
        height=height,
    )

    container = w.VBox(
        [fig],
        layout=w.Layout(
            width=f"{width}px", height=f"{height}px",
            margin="20px 14px 20px",
            border=f"2px solid {_THEME['card_border']}",
            overflow="hidden",
        ))
    
    container.add_class("border-radius")

    return container, fig, index_to_trace


def _build_mini_radar(scores: pd.DataFrame, idx: int, axes: Sequence[str], thetas: Sequence[str], cell_h: int, cell_w: int) -> Tuple[w.VBox, go.FigureWidget]:
    row = scores.loc[idx]
    rs = [_as_float(scores.at[idx, a]) for a in axes]
    r_closed = rs + [rs[0]]

    mini = go.FigureWidget(data=[go.Scatterpolar(
            r=r_closed, theta=thetas, mode="lines",
            line=dict(color=scores["color"][idx], width=2),
            fill="toself", fillcolor=scores["fill_color"][idx],
            name=""
        )])
    
    mini._orig_line = scores["color"][idx]
    mini._orig_fill = scores["fill_color"][idx]
    mini._config = {'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False}

    plot_h = max(24, int(cell_h))
    plot_w = max(24, int(cell_w))

    mini.update_layout(
        uirevision="keep",
        paper_bgcolor=_THEME["card_bg"],
        dragmode=False, showlegend=False,
        autosize=False, width=plot_w, height=plot_h,
        margin=dict(l=0, r=0, t=0, b=0),
        polar=dict(
            domain=dict(x=[0.13, 1 - 0.13], y=[0.22, 1 - 0.22]),
            radialaxis=dict(range=[0, 1], gridcolor=_THEME["grid"], showticklabels=False, showline=False, ticks=""),
            angularaxis=dict(
                tickmode="array",
                tickvals=axes,
                ticktext=[lbl.replace(" ", "<br>") for lbl in axes],
                tickfont=dict(color=_THEME["text"], size=10),
                gridcolor=_THEME["grid"], linecolor=_THEME["grid"], rotation=90, layer="below traces", showgrid=False),
            gridshape='circular',
        ),
    )

    mini.add_annotation(
        x=0.5, y=0.02, xref="paper", yref="paper",
        text=f"{row['Name']}",
        showarrow=False, align="center",
        font=dict(size=13, color=_THEME["text"], family="Arial Black, Segoe UI Semibold, Inter, Arial, sans-serif"),
        bgcolor=_THEME["card_bg"],
        borderpad=0
    )

    card = w.VBox(
        [mini],
        layout=w.Layout(
            width="100%", height=f"{cell_h}px",
            min_width="0",
            border=f"2px solid {_THEME['card_border']}",
            align_items="stretch",
            justify_content="center",
            overflow="hidden", padding="0",
            background_color=_THEME["card_bg"],
        ))
    card.add_class("border-radius")
    card.add_class("mini-card")

    return card, mini


def _build_radar_grid(scores: pd.DataFrame, height: int, width: int, axes: List[str], gap: int = 10) -> Tuple[w.GridBox, Dict[int, go.FigureWidget], Dict[int, w.VBox], List[str]]:
    cell_h = (height - gap * 2) // 3
    cell_w = (width - gap * 2) // 3

    minis_by_idx, cards_by_idx, children = {}, {}, []

    area_ids = [f"a{i}" for i in range(1, 10)]
    template = '"a1 a2 a3" "a4 a5 a6" "a7 a8 a9"'

    for pos, idx in enumerate(scores.index[:9]):
        card, mini = _build_mini_radar(scores, idx, axes, axes + [axes[0]], cell_h, cell_w)
        card.layout.grid_area = area_ids[pos]
        children.append(card)
        minis_by_idx[idx] = mini
        cards_by_idx[idx] = card

    grid = w.GridBox(
        children=tuple(children),
        layout=w.Layout(
            grid_template_areas=template,
            grid_template_columns="repeat(3, 1fr)",
            grid_auto_rows=f"{cell_h}px",
            grid_gap=f"{gap}px",
            align_items="stretch",
            overflow="hidden",
            width=f"{width}px",
            height=f"{height}px",
        )
    )

    return grid, minis_by_idx, cards_by_idx, area_ids


# ---------- Interaction ----------

def _apply_state(idx: int, active: bool, big_fig: go.FigureWidget, index_to_trace: Dict[int, int], mini: go.FigureWidget, card: w.VBox) -> None:
    """Toggle active/inactive styles for big and mini radar."""

    tnum = index_to_trace.get(idx)
    if tnum is not None:
        big_fig.data[tnum].visible = True if active else "legendonly"

    line = getattr(mini, "_orig_line", _THEME["muted_line"]) if active else _THEME["muted_line"]
    fill = getattr(mini, "_orig_fill", _THEME["muted_fill"]) if active else _THEME["muted_fill"]
    tcolor = _THEME["text"] if active else _THEME["muted_text"]
    bg = _THEME["card_bg"] if active else _THEME["mini_bg_inactive"]
    border = f"2px solid {_THEME['card_border']}" if active else "2px dashed rgba(0,0,0,0.28)"

    with mini.batch_update():
        trace = cast(go.Scatterpolar, mini.data[0])
        trace.update(
            line=dict(color=line),
            fillcolor=fill,
            opacity=(1.0 if active else 0.55),
        )

        mini.layout.paper_bgcolor = bg
        mini.layout.plot_bgcolor = bg
        mini.layout.polar.bgcolor = bg
        mini.layout.polar.angularaxis.tickfont.color = tcolor

    card.layout.background_color = bg
    card.layout.border = border


def _sort_value(scores: pd.DataFrame, i: int, col: str) -> float:
    return _as_float(scores.at[i, col])


def _compute_order(kind:str, scores:pd.DataFrame, original_order: List[int]) -> List[int]:
    key = _SORT_KEYS.get(kind)
    if key is None:
        return list(original_order)
    return sorted(original_order, key=lambda i: _sort_value(scores, i, key), reverse=True)



def build_radar_dashboard(scores: pd.DataFrame, height: int, width: int, names: List[str], on_toggle: Optional[Callable[[int, bool], None]] = None) -> w.VBox:
    # build main large spider
    big_card, big_fig, index_to_trace = _build_big_radar(scores, list(scores.index), height, width, names)

    # build 3x3 minis
    grid_height = max(60, height - _CONSTANTS["sort_bar_height"] - _CONSTANTS["sort_bar_gap"])
    grid, minis_by_idx, cards_by_idx, area_ids = _build_radar_grid(scores, grid_height, width, names)

    # which minis are active initially
    active: Dict[int, bool] = {idx: True for idx in minis_by_idx.keys()}
    original_order: List[int] = list(scores.index[:9])

    # sorting UI
    sort_dropdown = w.Dropdown(options=_SORT_LABELS, value="original", layout=w.Layout(width=f"{width}px"))
    sort_label = w.HTML("<span class='sort-label'><b>Auswahl:</b></span>")
    sort_bar = w.HBox(
        [sort_label, sort_dropdown],
        layout=w.Layout(
            width=f"{width}px",
            height=f"{_CONSTANTS['sort_bar_height']}px",
            align_items="center",
            margin=f"0 0 {_CONSTANTS['sort_bar_gap']}px 0",
        )
    )

    # holding the sort bar + mini grid
    grid_card = w.VBox(
        [sort_bar, grid], 
        layout=w.Layout(
            height=f"{height}px",
            width=f"{width}px",
            margin="20px 14px 20px",
        ))
    
    def _apply_sort(kind: str) -> None:
        order = _compute_order(kind, scores, original_order)

        for slot, idx in enumerate(order):
            cards_by_idx[idx].layout.grid_area = area_ids[slot]

        for idx in order:
            card = cards_by_idx[idx]
            live_mini = card.children[0]
            _apply_state(idx, active[idx], big_fig, index_to_trace, live_mini, card)

    sort_dropdown.observe(lambda ch: _apply_sort(ch["new"]) if ch["name"] == "value" else None, names="value")

    # wire up click/hover interactions for each mini card
    for idx in minis_by_idx.keys():
        card = cards_by_idx[idx]

        clicker = ev.Event(source=card, watched_events=["click"])
        def _on_click(event, _idx=idx, _card=card):
            new_state = not active[_idx]
            active[_idx] = new_state
            live_mini = _card.children[0]
            _apply_state(_idx, new_state, big_fig, index_to_trace, live_mini, _card)
            if on_toggle:
                on_toggle(_idx, new_state)
        clicker.on_dom_event(_on_click)

        hoverer = ev.Event(source=card, watched_events=["mouseenter", "mouseleave"])
        def _on_hover(evt, _idx=idx, _card=card):
            et = evt.get("type", "")
            if et == "mouseenter":
                _card.layout.border = f"2px solid {_THEME['hover_border']}"
            elif et == "mouseleave":
                live_mini = _card.children[0]
                _apply_state(_idx, active[_idx], big_fig, index_to_trace, live_mini, _card)
        hoverer.on_dom_event(_on_hover)

    # initial paint of styles
    for idx in minis_by_idx.keys():
        _apply_state(idx, True, big_fig, index_to_trace, minis_by_idx[idx], cards_by_idx[idx])

    # NEW: required title, left aligned, bold
    title_html = w.HTML(
        (
            "<div style='"
            "font-weight:800;"
            "text-align:left;"
            "margin:0 0 6px 0;"
            "width:100%;"
            "font-size:18px;"
            "'>"
            "Vergleich der Seiltrassenmodelle"
            "</div>"
        ),
        layout=w.Layout(width="100%")
    )

    # wrap everything, include title first
    container = w.VBox(
        [title_html, big_card, grid_card, _BORDER_RADIUS_CSS, _POINTER_CSS, _DROPDOWN_CSS],
        layout=w.Layout(
            width="100%",
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",        # keep left alignment
            justify_content="center",
        )
    )
    container.add_class("app-scope")
    
    return container
