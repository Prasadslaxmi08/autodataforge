"""NavigationPanel — the left sidebar (Phase 18 redesign).

A two-group tree: a always-visible **Workspace** group (the linear product flow —
Projects → Annotation → Review → Export) and a collapsible **Developer Tools** group
holding the advanced engineering modules, collapsed by default so a first-time user
sees only the main flow.

The shell contract is unchanged: the panel still emits `navigated(page_name)` and
still exposes `select(page_name)`, so `MainWindow` treats it exactly as before. Leaf
items carry their target *page name* (which may differ from the friendly label, e.g.
"Annotation" → "Annotation Pipeline") in a data role.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

_PAGE_ROLE = Qt.ItemDataRole.UserRole

# (group title, [(label, page_name), ...], expanded_by_default)
NAV_GROUPS: list[tuple[str, list[tuple[str, str]], bool]] = [
    ("WORKSPACE", [
        ("Projects", "Projects"),
        ("Annotation", "Annotation"),
        ("Review", "VLM Verification"),
        ("Export", "Export"),
    ], True),
    ("DEVELOPER TOOLS", [
        ("Planner", "Planner"),
        ("Annotation Pipeline", "Annotation Pipeline"),
        ("Dataset Intelligence", "AI Dataset Analyst"),
        ("Knowledge Center", "Engineering Memory"),
        ("Benchmark Center", "Benchmark Center"),
        ("Dashboard", "Dashboard"),
        ("Reports", "Reports"),
        ("Settings", "Settings"),
    ], False),
]

# Flat list of every navigable page name — kept for the shell/tests that enumerate
# reachable modules.
NAV_ITEMS: list[str] = [page for _title, leaves, _exp in NAV_GROUPS for _label, page in leaves]


class NavigationPanel(QTreeWidget):
    navigated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Nav")
        self.setFixedWidth(220)
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setIconSize(QSize(16, 16))
        self.setAnimated(True)

        self._by_page: dict[str, QTreeWidgetItem] = {}
        for title, leaves, expanded in NAV_GROUPS:
            group = QTreeWidgetItem([title])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)  # selectable off: it's a header
            group.setData(0, _PAGE_ROLE, None)
            font = group.font(0)
            font.setPointSize(max(1, font.pointSize() - 1))
            font.setBold(True)
            group.setFont(0, font)
            self.addTopLevelItem(group)
            for label, page in leaves:
                leaf = QTreeWidgetItem([label])
                leaf.setData(0, _PAGE_ROLE, page)
                group.addChild(leaf)
                self._by_page[page] = leaf
            group.setExpanded(expanded)

        self.itemClicked.connect(self._on_item)
        self.currentItemChanged.connect(lambda cur, _prev: self._emit(cur))

    # A header click toggles its own expansion; a leaf click navigates.
    def _on_item(self, item: QTreeWidgetItem, _col: int) -> None:
        if item.data(0, _PAGE_ROLE) is None:  # group header
            item.setExpanded(not item.isExpanded())

    def _emit(self, item: QTreeWidgetItem | None) -> None:
        if item is not None:
            page = item.data(0, _PAGE_ROLE)
            if page:
                self.navigated.emit(page)

    def select(self, page_name: str) -> None:
        item = self._by_page.get(page_name)
        if item is None:
            # Fall back to the first leaf (Projects) so the shell always has a page.
            item = self._by_page.get(NAV_ITEMS[0])
        if item is not None:
            item.parent().setExpanded(True)
            self.setCurrentItem(item)
