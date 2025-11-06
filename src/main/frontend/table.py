from turtle import color
import ipywidgets as w
from typing import List, Optional, Callable  # ← added Callable

_THEME = {
    "card_bg": "rgb(241, 248, 241)",
    "card_border": "#94b48a",
    "text": "#0f2010",
    "hover_border": "#48723b",
}

_TABLE_CSS = w.HTML(f"""
<style>
  .my-table {{
    border-radius: 8px;                 /* outer frame can stay rounded */
    background-color: #94b48a;          /* used as the 'gridline' color */
  }}

  /* Cell backgrounds (zebra) */
  .change-background {{ background-color: rgb(241, 248, 241); }}
  .no-background     {{ background-color: #ffffff; }}

  /* Labels */
  .tbl-label {{
    white-space: normal !important;
    overflow-wrap: anywhere !important;
  }}
  /* Header: same as cells, just bigger + bold */
  .header .tbl-label {{
    font-weight: 700 !important;
    font-size: 13.5px !important;
    color: #0f2010 !important;          /* same text color as body */
    text-align: center !important;
    width: 100% !important;
    display: block;
  }}

  /* Row wrapper = the "row" lines; sharp edges, no extra padding */
  .row-card {{
    border: 0;
    border-radius: 0;                    /* sharp edges per request */
    margin: 0;                           /* no extra row spacing here */
    padding: 0;                          /* keep rows compact */
  }}
  /* Make the gap between rows exactly 1px (same as cell gap) */
  .row-card + .row-card {{ margin-top: 1px; }}

  /* Selected row border emphasis (kept subtle) */
  .row-card--selected > .change-background,
  .row-card--selected > .no-background {{
        background-color: #fff3b0 !important; /* soft yellow */
    }}

  /* Buttons (unchanged good look) */
  .tbl-btn {{
    border-radius: 14px !important;
    border: 2px solid {_THEME['card_border']} !important;
    background-color: {_THEME['card_bg']} !important;
    color: {_THEME['text']} !important;
    font-weight: 600 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 34px !important;
    padding: 0 12px !important;
    line-height: 1 !important;
    cursor: pointer !important;
    transition: transform .06s ease, box-shadow .06s ease, border-color .06s ease, background-color .06s ease;
    user-select: none !important;
  }}
  .tbl-btn:hover {{ border-color: {_THEME['hover_border']} !important; transform: translateY(-1px) !important; }}
  .tbl-btn:focus-visible {{ outline: 2px solid #6aa86a !important; outline-offset: 2px !important; }}
  .tbl-btn--selected {{
    background: #2f6f3e !important;
    color: #ffffff !important;
    border-color: #225c2f !important;
    box-shadow: 0 0 0 3px rgba(47,111,62,.22);
  }}
  .row-card, .row-card * {{
    overflow-y: visible !important;
  }}
</style>
""", layout=w.Layout(display="none"))


def _build_cell(text: str, isColor: bool = False, isHeader: bool = False) -> w.VBox:
    label = w.Label(str(text), layout=w.Layout(margin="0", height="auto", width="auto"))
    label.add_class("tbl-label")

    box = w.Box(
        [label],
        layout=w.Layout(
            width="auto", height="auto",
            display="flex", align_items="center", justify_content="center",
            padding="4px 6px",
        )
    )

    if isHeader:
        # Header looks like cells, just bold/bigger via CSS class
        box.add_class("header")
        box.add_class("no-background")      # same background as a normal cell
    else:
        box.add_class("change-background" if isColor else "no-background")

    return box


class Table:
    """
    Simple ipywidgets table:
    - Renders a header row + data rows in a CSS-styled GridBox.
    - Supports row highlight.
    - Supports visibility toggling.
    - Optional title shown above the table. If `title` is None/empty, no title.

    Public methods:
      getWidget() -> root widget
      highlight_row(i) / clear_highlight()
      update_data(new_rows)
      set_visibility(bool)
    """
    def __init__(
        self,
        headers: List[str],
        data: List[List[str]],
        width: int,
        is_visible: bool = True,
        gap: int = 1,
        title: Optional[str] = None,
        action_label: Optional[str] = None,
        on_action: Optional[Callable[[int], None]] = None,
    ):
        self.headers = headers
        # action column toggling (kept fully optional so other tables are unchanged)
        self._has_action = (action_label is not None) and callable(on_action)
        self._action_label = action_label
        self._on_action = on_action
        self._selected_button: int = -1
        self._action_buttons: List[w.Button] = []
        self._highlighted_row: int | None = None
        self._row_wrappers: List[w.GridBox] = []
        self._header_wrapper: Optional[w.GridBox] = None

        # columns = headers (+1 if action column enabled)
        self.cols = len(headers) + (1 if self._has_action else 0)
        self.width = width
        self.is_visible = is_visible
        self._selected: int = -1
        self._muted: set[int] = set()

        # Build the main grid (header + body)
        self.grid = w.VBox(
            children=self.build_table(data),
            layout=w.Layout(
                width="100%",
                min_width="0",
                max_width=f"{width}px",
                border="2px solid #94b48a",
                gap="1px",
            )
        )
        self._scroll = w.Box(
            [self.grid],
            layout=w.Layout(
                width="100%",
                overflow_x="auto",
                overflow_y="hidden",
            )
        )
        self.grid.add_class("my-table")

        # Optional title block
        if title is not None and str(title).strip() != "":
            self._title_widget = w.HTML(
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
        else:
            self._title_widget = w.HTML("", layout=w.Layout(display="none"))

        # Root container: title (if any), then table, then CSS injector
        self.root = w.VBox(
            [self._title_widget, self._scroll, _TABLE_CSS],
            layout=w.Layout(
                width="100%",
                max_width=f"{width}px",
                overflow="visible",
                align_items="flex-start",
            )
        )

        # Apply initial visibility
        self.set_visibility(is_visible)

    # ---------------- public API ----------------
    def _set_selected_button(self, idx: int) -> None:
        if not self._has_action:
            return
        
        if self._selected_button != -1 and self._selected_button < len(self._action_buttons):
            try:
                self._action_buttons[self._selected_button].remove_class("tbl-btn--selected")
            except Exception:
                pass
        
        self._selected_button = idx
        if 0 <= idx < len(self._action_buttons):
            try:
                self._action_buttons[idx].add_class("tbl-btn--selected")
            except Exception:
                pass

    def getWidget(self) -> w.Widget:
        return self.root

    def clear_highlight(self) -> None:
        if getattr(self, "_highlighted_row", None) is not None:
            prev = self._highlighted_row
            if prev is not None and 0 <= prev < len(self._row_wrappers):
                try:
                    self._row_wrappers[prev].remove_class("row-card--selected")
                except Exception:
                    pass
        self._highlighted_row = None

        # clear button selection
        self._set_selected_button(-1)

    def highlight_row(self, idx: int) -> None:
        """Highlight the given row and sync the Auswählen button selection."""
        if idx == -1:
            self.clear_highlight()
            return
        if not (0 <= idx < len(self._row_wrappers)):
            return

        # remove old
        if self._highlighted_row is not None and 0 <= self._highlighted_row < len(self._row_wrappers):
            try:
                self._row_wrappers[self._highlighted_row].remove_class("row-card--selected")
            except Exception:
                pass

        # apply new
        try:
            self._row_wrappers[idx].add_class("row-card--selected")
        except Exception:
            pass
        self._highlighted_row = idx

        # keep button in sync
        self._set_selected_button(idx)

    def update_data(self, new_data: List[List[str]]) -> None:
        """
        Replace all rows with new_data (same number of cols as headers).
        Clears selection highlight.
        """
        self.grid.children = tuple(self.build_table(new_data))
        self._selected = -1

        for idx in list(self._muted):
            if idx < len(self._row_cells):
                for cell in self._row_cells[idx]:
                    cell.add_class("muted-row")
            else:
                self._muted.discard(idx)

    def set_visibility(self, is_visible: bool) -> None:
        """
        Show/hide the whole table wrapper.
        """
        self.is_visible = is_visible
        self.root.layout.display = "block" if is_visible else "none"

    # ---------------- internals ----------------

    def build_table(self, data: List[List[str]]) -> List[w.Widget]:
        """
        Build header row (self.headers) and data rows (data),
        remember each cell widget so we can highlight rows later.
        """
        # validate column counts (against headers only; action column is internal)
        for row in data:
            if len(row) != len(self.headers):
                raise ValueError(
                    "All rows must have the same number of columns as headers"
                )

        self.data = data
        self._row_cells = []
        self._all_cells = []
        self._row_wrappers = []

        children: List[w.Widget] = []

        # ---------- Header wrapper ----------
        header_cells: List[w.Box] = []
        for h in self.headers:
            c = _build_cell(h, isColor=True, isHeader=True)
            header_cells.append(c)
            self._all_cells.append(c)
        if self._has_action:
            c = _build_cell("", isColor=True, isHeader=True)
            header_cells.append(c)
            self._all_cells.append(c)

        header_grid = w.GridBox(
            children=tuple(header_cells),
            layout=w.Layout(
                width="100%",
                grid_template_columns=f"repeat({self.cols}, minmax(0, 1fr))",
                grid_gap="1px",
            )
        )
        header_grid.add_class("row-card")
        self._header_wrapper = header_grid
        children.append(header_grid)

        # ---------- Body rows as wrappers ----------
        self._action_buttons = []

        for i, row in enumerate(data):
            row_cells: List[w.Box] = []

            # per-cell boxes
            for item in row:
                cell = _build_cell(item, isColor=(i % 2 == 0))
                row_cells.append(cell)
                self._all_cells.append(cell)

            # optional button cell (kept, unchanged)
            if self._has_action:
                btn = w.Button(description=self._action_label or "Auswählen")
                btn.add_class("tbl-btn")
                btn.layout = w.Layout(height="36px")

                btn_box = w.Box(
                    [btn],
                    layout=w.Layout(
                        width="auto", height="auto",
                        display="flex", align_items="center",
                        justify_content="center", padding="6px 8px",
                    ),
                )
                # keep zebra feel inside the row
                (btn_box.add_class("change-background") if i % 2 == 0 else btn_box.add_class("no-background"))

                def _make_cb(idx: int, button: w.Button):
                    def _cb(_):
                        # sync the button + row card selection
                        self._set_selected_button(idx)
                        self.highlight_row(idx)   # will toggle the row-card border
                        try:
                            if callable(self._on_action):
                                self._on_action(idx)
                        except Exception:
                            pass
                    return _cb

                btn.on_click(_make_cb(i, btn))
                self._action_buttons.append(btn)
                row_cells.append(btn_box)
                self._all_cells.append(btn_box)

            # build row wrapper as a grid with N columns
            row_grid = w.GridBox(
                children=tuple(row_cells),
                layout=w.Layout(
                    width="100%",
                    grid_template_columns=f"repeat({self.cols}, minmax(0, 1fr))",
                    grid_gap="1px",
                )
            )
            row_grid.add_class("row-card")
            self._row_wrappers.append(row_grid)
            children.append(row_grid)

            self._row_cells.append(row_cells)

        return children
    
    def set_row_muted(self, index: int, muted: bool) -> None:
        if index < 0 or index >= len(self._row_cells):
            return
        if muted:
            self._muted.add(index)
            for cell in self._row_cells[index]:
                cell.add_class("muted-row")
        else:
            if index in self._muted:
                self._muted.remove(index)
            for cell in self._row_cells[index]:
                cell.remove_class("muted-row")

    def mute_row(self, index: int) -> None:
        self.set_row_muted(index, True)

    def unmute_row(self, index: int) -> None:
        self.set_row_muted(index, False)

    def clear_button_selection(self) -> None:
        if self._selected_button != -1 and self._selected_button < len(self._action_buttons):
            self._action_buttons[self._selected_button].remove_class("tbl-btn--selected")
        self._selected_button = -1
