"""PlaceholderPage (Phase 11).

A complete, styled layout with placeholder content for modules that will be wired
to the backend in later phases (Planner, Annotation, VLM, Analyst, Memory,
Benchmarks, Reports, Settings). It shows the *intended* structure — titled section
cards — so the shell already reads as the finished tool, not a blank stub. No
backend calls, by design (per the phase brief: only Dashboard and Dataset Manager
are functional now).
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from vds.gui.pages.base import Page
from vds.gui.widgets.common import Card, label


def make_placeholder(page_name: str, subtitle: str, sections: list[tuple[str, str]]) -> type[Page]:
    """Build a Page subclass rendering `sections` as [(title, description)] cards."""

    class _Placeholder(Page):
        name = page_name

        def __init__(self) -> None:
            super().__init__()
            root = QVBoxLayout(self)
            root.setContentsMargins(20, 18, 20, 18)
            root.setSpacing(12)
            root.addWidget(label(page_name, "H1"))
            sub = label(subtitle, "Muted", wrap=True)
            root.addWidget(sub)
            root.addWidget(label("Integration planned for a later phase.", "Badge"))

            grid = QGridLayout()
            grid.setSpacing(12)
            for i, (title, desc) in enumerate(sections):
                card = Card(title)
                card.add(label(desc, "Muted", wrap=True))
                grid.addWidget(card, i // 2, i % 2)
            holder = QWidget()
            holder.setLayout(grid)
            root.addWidget(holder)
            root.addStretch(1)

        def context(self) -> tuple[str, list[tuple[str, str]]]:
            return (page_name, [("Status", "Placeholder"),
                                ("Backend", "Not yet integrated")])

    return _Placeholder


# The eight placeholder modules, each with a described intended layout.
PLACEHOLDER_SPECS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "Reports": ("Browse generated performance, analyst, and engineering reports.",
                [("Report Browser", "All generated markdown reports."),
                 ("Preview", "Rendered report content."),
                 ("Export", "Save or share a report."),
                 ("History", "Reports across projects and runs.")]),
    "Settings": ("Configure provider/model, GPU budget, export defaults, and appearance.",
                 [("Provider & Model", "LLM provider selection and model id."),
                  ("Compute", "GPU device and VRAM budget."),
                  ("Export Defaults", "Default annotation export format."),
                  ("Appearance", "Theme and layout preferences.")]),
}
