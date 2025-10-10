import ipywidgets as w
from typing import Callable, Optional

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

        self.result_dropdown = w.Dropdown(
            options=results_options,
            value=None,
            description="Select Result:",
            disabled=False,
            style={'description_width': 'initial'},
            layout=w.Layout(width="300px")
        )

        self.result_dropdown.observe(self._handle_change, names="value")

    def _handle_change(self, change):
        if change.get("name") == "value":
            if callable(self._on_select):
                self._on_select(change["new"])

    def on_select(self, fn: Callable[[Optional[int]], None]) -> None:
        self._on_select = fn

    def get_widget(self) -> w.Widget:
        return self.result_dropdown

        
        
    