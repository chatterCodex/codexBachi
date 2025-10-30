# interface.py
import pandas as pd
import ipywidgets as w

from src.main.frontend.map import Map
from src.main.frontend.result_selector import ResultSelector
from src.main.frontend.radar_chart import build_radar_dashboard
from src.main.frontend.table import Table

# NEW: pull the single-entry factory that precomputes everything
from src.main.frontend.data_prep import build_viz_data

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

def build_interface_with_viz_data(vd, results_df) -> w.VBox:
    map_component = Map(vd.map, "Seiltrassen Karte")

    # Radar chart
    scores = vd.make_radar_scores(_NAMES["axes"])
    radar_chart = build_radar_dashboard(scores, 600, 750, _NAMES["axes"])

    # Overview table (no title by design)
    overview_table = Table(_NAMES["table_overview_headers"], vd.overview_rows, 1500, title=None)

    # Detail tables (now with titles)
    selected_table = Table(_NAMES["table_selected_headers"], [], 950, is_visible=False, title="Aktivierte Seiltrassen")
    anchor_table   = Table(_NAMES["table_anchor_headers"],  [], 450, is_visible=False, title="Endmast Informationen")

    # Result selector → uses vd (instant switching, no recompute)
    selector = ResultSelector(
        num_results=len(results_df),
        on_select=lambda idx: _on_select_with_vd(
            idx, vd, results_df, selected_table, anchor_table, map_component, overview_table
        ),
    )

    toolbar = w.HBox(
        [selector.get_widget()],
        layout=w.Layout(width="100%", max_width="1500px", align_items="center", margin="5px 0"),
    )

    sel_widget  = selected_table.getWidget()
    anch_widget = anchor_table.getWidget()

    left_wrap  = w.Box([sel_widget],  layout=w.Layout(margin="0 10px 0 0", flex="1 1 auto"))
    right_wrap = w.Box([anch_widget], layout=w.Layout(margin="0 0 0 10px", flex="1 1 auto"))

    details_row = w.HBox(
        [left_wrap, right_wrap],
        layout=w.Layout(
            width="100%",
            align_items="flex-start",
            margin="20px 0",
            overflow="auto",
        ),
    )

    ui = w.VBox(
        [
            map_component.get_map_widget(),
            toolbar,
            radar_chart,
            overview_table.getWidget(),
            details_row,
        ],
        layout=w.Layout(align_items="center", gap="16px", width="100%", padding="16px 20px"),
    )

    return ui



def build_interface(forest_area_3, model_list, results_df: pd.DataFrame) -> w.VBox:
    # Build once: distances, per-model layouts, union layout, stable map payload, overview rows, etc.
    vd = build_viz_data(forest_area_3, model_list, results_df)

    # Map (stable hover: length + fixed volume)
    map_component = Map(vd.map, "Seiltrassen Karte")

    # Radar chart
    scores = vd.make_radar_scores(_NAMES["axes"])
    radar_chart = build_radar_dashboard(scores, 600, 750, _NAMES["axes"])

    # Overview table (no title by design)
    overview_table = Table(_NAMES["table_overview_headers"], vd.overview_rows, 1500, title=None)

    # Detail tables (now with titles)
    selected_table = Table(_NAMES["table_selected_headers"], [], 750, is_visible=False, title="Aktivierte Seiltrassen")
    anchor_table   = Table(_NAMES["table_anchor_headers"],  [], 450, is_visible=False, title="Endmast Informationen")

    # Result selector → uses vd (instant switching, no recompute)
    selector = ResultSelector(
        num_results=len(results_df),
        on_select=lambda idx: _on_select_with_vd(
            idx, vd, results_df, selected_table, anchor_table, map_component, overview_table
        ),
    )

    toolbar = w.HBox(
        [selector.get_widget()],
        layout=w.Layout(width="100%", max_width="1500px", align_items="center", margin="5px 0"),
    )

    sel_widget  = selected_table.getWidget()
    anch_widget = anchor_table.getWidget()

    left_wrap  = w.Box([sel_widget],  layout=w.Layout(margin="0 10px 0 0", flex="1 1 auto"))
    right_wrap = w.Box([anch_widget], layout=w.Layout(margin="0 0 0 10px", flex="1 1 auto"))

    details_row = w.HBox(
        [left_wrap, right_wrap],
        layout=w.Layout(
            width="100%",
            align_items="flex-start",
            margin="20px 0",
            overflow="auto",
        ),
    )

    ui = w.VBox(
        [
            map_component.get_map_widget(),
            toolbar,
            radar_chart,
            overview_table.getWidget(),
            details_row,
        ],
        layout=w.Layout(align_items="center", gap="16px", width="100%", padding="16px 20px"),
    )

    return ui


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
