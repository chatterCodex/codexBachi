from tkinter import font
import ipywidgets as w
from typing import List

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
  .selected-background {
    background: #fde68a;
  }
  .tbl-label {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
  }
</style>
""", layout=w.Layout(display="none"))

def _build_cell(text: str, isColor: bool = False, isHeader: bool = False) -> w.VBox:
    label = w.Label(str(text), layout=w.Layout(margin="0", height="auto", width="auto"))
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
        box.add_class("header")
    if isColor:
        box.add_class("change-background")
    else:
        box.add_class("no-background")

    return box

class Table:
    def __init__(self, headers: List[str], data: List[List[str]], width: int, gap: int = 1):
        self.headers = headers
        self.data = data
        self.cols = len(headers)

        for d in data:
            if len(d) != self.cols:
                raise ValueError("All rows must have the same number of columns as headers")
            
        self._row_cells: List[List[w.Box]] = []
        self._all_cells: List[w.Box] = []

        cells: List[w.Box] = []

        for h in headers:
            cell = _build_cell(h, True, True)
            cells.append(cell)
            self._all_cells.append(cell)

        for i, row in enumerate(data):
            row_cells: List[w.Box] = []
            for item in row:
                cell = _build_cell(item, (i % 2 == 1))
                cells.append(cell)
                row_cells.append(cell)
                self._all_cells.append(cell)
            self._row_cells.append(row_cells)

        self.grid = w.GridBox(
            children=tuple(cells),
            layout=w.Layout(
                width=f"{width}px",
                height="auto",
                grid_template_columns=f"repeat({self.cols}, minmax(0, 1fr))",
                grid_template_rows=f"auto repeat({len(self._row_cells)}, auto)",
                grid_gap=f"{gap}px",
                align_items="stretch",
                border="2px solid #94b48a",
            )
        )

        self.grid.add_class("my-table")
        self.root = w.VBox([self.grid, _TABLE_CSS], layout=w.Layout(width=f"{width}px"))

        self._selected: int = -1

    def getWidget(self) -> w.Widget:
        return self.root
    
    def clear_highlight(self):
        if self._selected == -1:
            return
        
        for cell in self._row_cells[self._selected]:
            cell.remove_class("selected-background")

    def highlight_row(self, index: int):
        if index < -1 or index >= len(self._row_cells):
            raise IndexError(f"Row index {index} is out of range 0..{len(self._row_cells)-1}")
        
        self.clear_highlight()

        if index != -1:
            for cell in self._row_cells[index]:
                cell.add_class("selected-background")

        self._selected = index