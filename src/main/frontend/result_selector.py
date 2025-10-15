import ipywidgets as w
from typing import Callable, Optional

_THEME = {
    "card_bg": "rgb(241, 248, 241)",
    "card_border": "#94b48a",
    "text": "#0f2010",

    "hover_border": "#48723b",
}


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

class ResultSelector:
    def __init__(
            self,
            num_results: int = 9,
            on_select: Optional[Callable[[Optional[int]], None]] = None
    ):
        self.num_results = num_results
        self._on_select = on_select
        self._create_widgets()

    def _create_widgets(self):
        results_options = [("No selection", None)] + [
            (f"Result {i+1}", i) for i in range(self.num_results)
        ]

        result_dropdown =  w.Dropdown(
            options=results_options,
            value=None,
            description="Select Result:",
            disabled=False,
            style={'description_width': 'initial'},
            layout=w.Layout(width="300px")
        )

        result_dropdown.add_class("widget-dropdown")

        result_dropdown.observe(self._handle_change, names="value")

        self.widget = w.VBox([_DROPDOWN_CSS, result_dropdown])
        self.widget.add_class("app-scope")

    def _handle_change(self, change):
        if change.get("name") == "value":
            if callable(self._on_select):
                self._on_select(change["new"])

    def on_select(self, fn: Callable[[Optional[int]], None]) -> None:
        self._on_select = fn

    def get_widget(self) -> w.Widget:
        return self.widget

        
        
    