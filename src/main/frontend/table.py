import ipywidgets as w
from typing import List, Optional

_TABLE_CSS = w.HTML("""
<style>
  .header {
    font-weight: 700;
  }
  .my-table {
    border-radius: 8px;
    background-color: #94b48a;
  }
  .change-background {
    background-color: rgb(241, 248, 241);
  }
  .no-background {
    background-color: white;
  }
  /* Selected row highlight (overview table etc.) */
  .selected-background {
    background: #fde68a;
  }
  .tbl-label {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
  }
  .header .tbl-label {
    text-align: center !important;
    width: 100% !important;
    display: block;
  }
  .muted-row {
   background-color: #e5e7eb !important;  /* neutral light grey */
    }

    /* Make the text look muted too */
    .muted-row .tbl-label {
    color: #6b7280 !important; /* grey text */
    }

    /* If a row is both selected + muted, keep it readable */
    .selected-background.muted-row {
    background-color: #d1d5db !important;  /* slightly darker grey */
    }
</style>
""", layout=w.Layout(display="none"))


def _build_cell(text: str, isColor: bool = False, isHeader: bool = False) -> w.VBox:
    """
    Build a single table cell widget with correct background color
    and header styling.
    """
    label = w.Label(
        str(text),
        layout=w.Layout(margin="0", height="auto", width="auto")
    )
    label.add_class("tbl-label")

    box = w.Box(
        [label],
        layout=w.Layout(
            width="auto",
            height="auto",
            display="flex",
            align_items="center",
            justify_content="center",
            padding="6px 8px",
        )
    )

    if isHeader:
        label.layout.width = "100%"
        box.add_class("header")
    if isColor:
        box.add_class("change-background")
    else:
        box.add_class("no-background")

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
    ):
        self.headers = headers
        self.cols = len(headers)
        self.width = width
        self.is_visible = is_visible
        self._selected: int = -1
        self._muted: set[int] = set()

        # Build the main grid (header + body)
        self.grid = w.GridBox(
            children=tuple(self.build_table(data)),
            layout=w.Layout(
                width="auto",
                min_width=f"{width}px",
                height="auto",
                grid_template_columns=f"repeat({self.cols}, minmax(0, 1fr))",
                grid_template_rows=f"auto repeat({len(self._row_cells)}, auto)",
                grid_gap=f"{gap}px",
                align_items="stretch",
                border="2px solid #94b48a",
            )
        )
        self.grid.add_class("my-table")

        # Optional title block
        if title is not None and str(title).strip() != "":
            self._title_widget = w.HTML(
                (
                    "<div style='"
                    "font-weight:700;"         # thicker text
                    "text-align:left;"         # align text left
                    "margin:0 0 6px 0;"
                    "width:100%;"
                    "font-size:14px;"
                    "'>"
                    f"{title}"
                    "</div>"
                ),
                layout=w.Layout(width="auto")
            )
        else:
            self._title_widget = w.HTML("", layout=w.Layout(display="none"))

        # Root container: title (if any), then table, then CSS injector
        # align_items='flex-start' so the title hugs the left edge even if parent centers this VBox
        self.root = w.VBox(
            [self._title_widget, self.grid, _TABLE_CSS],
            layout=w.Layout(
                width="100%",
                overflow="visible",
                align_items="flex-start"
            )
        )

        # Apply initial visibility
        self.set_visibility(is_visible)

    # ---------------- public API ----------------

    def getWidget(self) -> w.Widget:
        return self.root

    def clear_highlight(self) -> None:
        """
        Remove highlight class from previously highlighted row.
        """
        if self._selected == -1:
            return
        for cell in self._row_cells[self._selected]:
            cell.remove_class("selected-background")
        self._selected = -1

    def highlight_row(self, index: int) -> None:
        """
        Highlight one specific row (0-based) in the table body.
        Pass -1 to clear.
        """
        if index < -1 or index >= len(self._row_cells):
            raise IndexError(
                f"Row index {index} is out of range 0..{len(self._row_cells)-1}"
            )

        self.clear_highlight()

        if index != -1:
            for cell in self._row_cells[index]:
                cell.add_class("selected-background")
            self._selected = index

    def update_data(self, new_data: List[List[str]]) -> None:
        """
        Replace all rows with new_data (same number of cols as headers).
        Clears selection highlight.
        """
        self.grid.children = tuple(self.build_table(new_data))
        self.grid.layout.grid_template_rows = f"auto repeat({len(self._row_cells)}, auto)"
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

    def build_table(self, data: List[List[str]]) -> List[w.Box]:
        """
        Build header row (self.headers) and data rows (data),
        remember each cell widget so we can highlight rows later.
        """
        # validate column counts
        for row in data:
            if len(row) != self.cols:
                raise ValueError(
                    "All rows must have the same number of columns as headers"
                )

        self.data = data
        self._row_cells: List[List[w.Box]] = []
        self._all_cells: List[w.Box] = []

        cells: List[w.Box] = []

        # Header row
        for h in self.headers:
            cell = _build_cell(h, isColor=True, isHeader=True)
            cells.append(cell)
            self._all_cells.append(cell)

        # Body rows
        for i, row in enumerate(data):
            row_cells: List[w.Box] = []
            for item in row:
                cell = _build_cell(item, isColor=(i % 2 == 1))
                cells.append(cell)
                row_cells.append(cell)
                self._all_cells.append(cell)
            self._row_cells.append(row_cells)

        return cells
    
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
