from pyexpat import model
import pandas as pd
import ipywidgets as w
from src.main.frontend.map import Map
from src.main.frontend.result_selector import ResultSelector
from src.main.frontend.radar_chart import build_radar_dashboard
from src.main.frontend.data_prep import make_radar_scores, get_overview_table_data, get_selected_table_data, get_anchor_table_data, prepare_map_data
from src.main.frontend.table import Table


_NAMES = {
    "axes": ["Ergonomische Optimierung", "Ökologische Optimierung", "Kosten Optimierung"],
    "title": "Vergleich der Seiltrassenmodelle",
    "table_overview_headers": [
        "Index",
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
    ]
}


def build_interface(forest_area_3, model_list, results_df: pd.DataFrame) -> w.VBox:
    scores = make_radar_scores(results_df, _NAMES["axes"])

    map_payload = prepare_map_data(forest_area_3, results_df)
    map_component = Map(map_payload)

    radar_chart = build_radar_dashboard(scores, 600, 700, _NAMES["axes"])

    overview_table = Table(_NAMES["table_overview_headers"], get_overview_table_data(forest_area_3, model_list, results_df), 1500)

    selected_table = Table(_NAMES["table_selected_headers"], [], 1500, is_visible=False)
    anchor_table = Table(_NAMES["table_anchor_headers"], [], 800, is_visible=False)

    selector = ResultSelector(num_results=len(results_df), on_select=lambda idx: _on_select(idx, forest_area_3, model_list, results_df, selected_table, anchor_table, map_component))

    # return w.VBox([radar_chart, overview_table.getWidget(), selector.get_widget(), selected_table.getWidget(), anchor_table.getWidget()], layout=w.Layout(align_items="center", gap="20px"))

    return w.VBox([map_component.get_map_widget(), selector.get_widget()], layout=w.Layout(align_items="center", gap="20px"))


def _on_select(selected_index, forest_area_3, model_list, results_df, selected_table: Table, anchor_table: Table, map_component: Map) -> None:
    if selected_index is None:
        selected_table.update_data([])
        selected_table.set_visibility(False)
        anchor_table.update_data([])
        anchor_table.set_visibility(False)
        map_component.update_map()
        return

    selected_lines = results_df.iloc[selected_index]["selected_lines"]

    sel_rows = get_selected_table_data(forest_area_3, model_list, results_df, selected_index)
    anch_rows = get_anchor_table_data(forest_area_3, model_list, results_df, selected_index)

    map_component.update_map(selected_index, selected_lines)

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






    
