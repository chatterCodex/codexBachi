# interface.py
import pandas as pd
import ipywidgets as w

from src.main.frontend.map import Map
from src.main.frontend.result_selector import ResultSelector
from src.main.frontend.radar_chart import build_radar_dashboard
from src.main.frontend.table import Table

# NEW: pull the single-entry factory that precomputes everything
from src.main.frontend.data_prep import build_viz_data

_APP_STYLE_HTML = """
<style>
  .app-shell {
    width: min(100%, 1320px);
    margin: 0 auto;
    padding: 16px 20px 32px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 20px;
  }

  .app-shell .section-block {
    width: 100%;
  }

  .app-shell .section-block.is-map {
    justify-content: center;
  }

  .app-shell .toolbar {
    width: 100%;
    display: flex;
    justify-content: flex-end;
  }

  .app-shell .toolbar .widget-box,
  .app-shell .toolbar .widget-hbox,
  .app-shell .toolbar .widget-vbox {
    width: min(100%, 520px);
  }

  .app-shell .details-row {
    display: flex;
    width: 100%;
    gap: 20px;
    align-items: stretch;
  }

  .app-shell .details-column {
    flex: 1 1 360px;
    min-width: 0;
    display: flex;
  }

  .app-shell .details-column > .widget-box,
  .app-shell .details-column > .widget-vbox {
    width: 100%;
  }

  @media (max-width: 1200px) {
    .app-shell {
      padding: 12px 16px 28px;
    }

    .app-shell .toolbar {
      justify-content: center;
    }
  }

  @media (max-width: 980px) {
    .app-shell .details-row {
      flex-direction: column;
    }

    .app-shell .details-column {
      flex-basis: auto;
    }
  }
</style>
"""


_NAMES = {
    "axes": ["Ergonomische Optimierung", "Ökologische Optimierung", "Kosten Optimierung"],
    "title": "Vergleich der Seiltrassenmodelle",
    "table_overview_headers": [
        "Modell",
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
    ],
    "table_selected_headers": [
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
    "table_anchor_headers": [
        "Seiltrassen Nummer",
        "BHD [cm]",
        "Height [m]",
        "X-Koordinate",
        "Y-Koordinate",
    ],
}

def _build_style_widget() -> w.HTML:
    return w.HTML(_APP_STYLE_HTML, layout=w.Layout(display="none"))

def build_interface_with_viz_data(vd, results_df) -> w.VBox:
    map_component = Map(vd.map, "Seiltrassen Karte")

    # Radar chart
    scores = vd.make_radar_scores(_NAMES["axes"])
    radar_chart = build_radar_dashboard(
        scores, 600, 750, _NAMES["axes"],
        on_toggle=lambda idx, active: overview_table.unmute_row(idx) if active else overview_table.mute_row(idx)
    )

    # Overview table (no title by design)
    # NEW: add per-row "Auswählen" buttons that sync the ResultSelector
    overview_table = Table(
        _NAMES["table_overview_headers"],
        vd.overview_rows,
        1500,
        title=None,
        action_label="Auswählen",
        on_action=lambda idx: selector.set_value(idx),  # selector is defined below; late-binding in lambda
    )

    # Detail tables (now with titles)
    selected_table = Table(_NAMES["table_selected_headers"], [], 550, is_visible=False, title="Aktivierte Seiltrassen")
    anchor_table   = Table(_NAMES["table_anchor_headers"],  [], 450, is_visible=False, title="Endmast Informationen")

    # Result selector → uses vd (instant switching, no recompute)
    selector = ResultSelector(
        num_results=len(results_df),
        on_select=lambda idx: _on_select_with_vd(
            idx, vd, results_df, selected_table, anchor_table, map_component, overview_table
        ),
    )

    selector_widget = selector.get_widget()
    selector_widget.layout.width = "100%"

    toolbar = w.HBox(
        [selector_widget],
        layout=w.Layout(width="100%", max_width="1500px", align_items="center", margin="5px 0"),
    )

    toolbar.add_class("toolbar")
    toolbar.layout.justify_content = "flex-end"

    sel_widget  = selected_table.getWidget()
    anch_widget = anchor_table.getWidget()

    sel_widget.layout.width = "100%"
    anch_widget.layout.width = "100%"

    left_wrap = w.Box(
        [sel_widget],
        layout=w.Layout(
            margin="0",
            flex="1 1 360px",
            min_width="0",
        ),
    )
    right_wrap = w.Box(
        [anch_widget],
        layout=w.Layout(
            margin="0",
            flex="1 1 360px",
            min_width="0",
        ),
    )
    left_wrap.add_class("details-column")
    right_wrap.add_class("details-column")

    details_row = w.HBox(
        [left_wrap, right_wrap],
        layout=w.Layout(
            width="100%",
            align_items="stretch",
            margin="20px 0",
            overflow="visible",
            flex_flow="row wrap",
            gap="20px",
        ),
    )

    details_row.add_class("details-row")
    details_row.add_class("section-block")

    map_widget = map_component.get_map_widget()
    map_widget.layout.width = "100%"
    map_widget.layout.justify_content = "center"
    map_widget.add_class("section-block")
    map_widget.add_class("is-map")

    radar_chart.layout.width = "100%"
    radar_chart.add_class("section-block")

    overview_widget = overview_table.getWidget()
    overview_widget.layout.width = "100%"
    overview_widget.add_class("section-block")

    ui = w.VBox(
        [
            map_widget,
            toolbar,
            radar_chart,
            overview_widget,
            details_row,
        ],
        layout=w.Layout(width="100%", align_items="stretch", gap="20px"),
    )

    ui.add_class("app-shell")

    style_widget = _build_style_widget()
    return w.VBox([style_widget, ui], layout=w.Layout(width="100%"))


def build_interface(forest_area_3, model_list, results_df: pd.DataFrame) -> w.VBox:
    # Build once: distances, per-model layouts, union layout, stable map payload, overview rows, etc.
    vd = build_viz_data(forest_area_3, model_list, results_df)

    # Map (stable hover: length + fixed volume)
    map_component = Map(vd.map, "Seiltrassen Karte")

    # Radar chart
    scores = vd.make_radar_scores(_NAMES["axes"])
    radar_chart = build_radar_dashboard(
        scores, 600, 750, _NAMES["axes"],
        on_toggle=lambda idx, active: overview_table.unmute_row(idx) if active else overview_table.mute_row(idx)
    )

    # Overview table (no title by design)
    # NEW: add per-row "Auswählen" buttons that sync the ResultSelector
    overview_table = Table(
        _NAMES["table_overview_headers"],
        vd.overview_rows,
        1500,
        title=None,
        action_label="Auswählen",
        on_action=lambda idx: selector.set_value(idx),  # selector defined below; safe due to late-binding
    )

    # Detail tables (now with titles)
    selected_table = Table(_NAMES["table_selected_headers"], [], 1000, is_visible=False, title="Aktivierte Seiltrassen")
    anchor_table   = Table(_NAMES["table_anchor_headers"],  [], 450, is_visible=False, title="Endmast Informationen")

    # Result selector → uses vd (instant switching, no recompute)
    selector = ResultSelector(
        num_results=len(results_df),
        on_select=lambda idx: _on_select_with_vd(
            idx, vd, results_df, selected_table, anchor_table, map_component, overview_table
        ),
    )

    selector_widget = selector.get_widget()
    selector_widget.layout.width = "100%"

    toolbar = w.HBox(
        [selector_widget],
        layout=w.Layout(width="100%", max_width="1500px", align_items="center", margin="5px 0"),
    )

    toolbar.add_class("toolbar")
    toolbar.layout.justify_content = "flex-end"

    sel_widget  = selected_table.getWidget()
    anch_widget = anchor_table.getWidget()

    sel_widget.layout.width = "100%"
    anch_widget.layout.width = "100%"

    left_wrap = w.Box(
        [sel_widget],
        layout=w.Layout(
            margin="0",
            flex="1 1 360px",
            min_width="0",
        ),
    )
    right_wrap = w.Box(
        [anch_widget],
        layout=w.Layout(
            margin="0",
            flex="1 1 360px",
            min_width="0",
        ),
    )
    left_wrap.add_class("details-column")
    right_wrap.add_class("details-column")

    details_row = w.HBox(
        [left_wrap, right_wrap],
        layout=w.Layout(
            width="100%",
            align_items="stretch",
            margin="20px 0",
            overflow="visible",
            flex_flow="row wrap",
            gap="20px",
        ),
    )

    details_row.add_class("details-row")
    details_row.add_class("section-block")

    map_widget = map_component.get_map_widget()
    map_widget.layout.width = "100%"
    map_widget.layout.justify_content = "center"
    map_widget.add_class("section-block")
    map_widget.add_class("is-map")

    radar_chart.layout.width = "100%"
    radar_chart.add_class("section-block")

    overview_widget = overview_table.getWidget()
    overview_widget.layout.width = "100%"
    overview_widget.add_class("section-block")

    ui = w.VBox(
        [
            map_widget,
            toolbar,
            radar_chart,
            overview_widget,
            details_row,
        ],
        layout=w.Layout(width="100%", align_items="stretch", gap="20px"),
    )
    ui.add_class("app-shell")

    style_widget = _build_style_widget()
    return w.VBox([style_widget, ui], layout=w.Layout(width="100%"))


def _on_select_with_vd(
    selected_index: int | None,
    vd,                       # VizData (precomputed)
    results_df: pd.DataFrame,
    selected_table: Table,
    anchor_table: Table,
    map_component: Map,
    overview_table: Table,
) -> None:
    """Selector callback wired to the precomputed VizData instance."""
    if selected_index is None:
        selected_table.update_data([])
        selected_table.set_visibility(False)
        anchor_table.update_data([])
        anchor_table.set_visibility(False)
        map_component.update_map()
        overview_table.highlight_row(-1)
        return

    # Highlight selected model in overview
    overview_table.clear_button_selection()
    overview_table.highlight_row(selected_index)

    # Get the display data for detail tables instantly (precomputed)
    sel_rows  = vd.selected_rows(selected_index)
    anch_rows = vd.anchor_rows(selected_index)

    # Update map coloring/selection:
    selected_lines = results_df.iloc[selected_index]["selected_lines"]
    map_component.update_map(selected_index, selected_lines)

    # Show/hide detail tables based on data presence
    if sel_rows:
        selected_table.update_data(sel_rows)
        selected_table.set_visibility(True)
    else:
        selected_table.update_data([])
        selected_table.set_visibility(False)

    if anch_rows:
        anchor_table.update_data(anch_rows)
        anchor_table.set_visibility(True)
    else:
        anchor_table.update_data([])
        anchor_table.set_visibility(False)
