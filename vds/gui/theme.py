"""ThemeManager — an engineering-IDE look (Phase 11).

Two flat, low-chroma themes (dark default, light) in the spirit of VS Code / Qt
Creator / JetBrains: muted surfaces, a single accent, compact controls, no
oversized consumer chrome. Themes are plain QSS applied to the whole QApplication;
switching restyles live. The chosen theme is persisted via GuiSettings.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

ACCENT = "#3d7eff"

_DARK = {
    "bg": "#1e1f22", "surface": "#2b2d30", "surface2": "#232427", "border": "#3a3d41",
    "text": "#d7dae0", "muted": "#8b9096", "accent": ACCENT, "accent_text": "#ffffff",
    "sel": "#2f5199",
}
_LIGHT = {
    "bg": "#f3f4f6", "surface": "#ffffff", "surface2": "#eceef1", "border": "#d0d3d9",
    "text": "#22252a", "muted": "#6b7178", "accent": ACCENT, "accent_text": "#ffffff",
    "sel": "#cfe0ff",
}

_QSS = """
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px; }}
QMainWindow, QWidget {{ background: {bg}; color: {text}; }}
QFrame#Card, QFrame#Panel {{
    background: {surface}; border: 1px solid {border}; border-radius: 8px;
}}
QLabel#Hero {{ font-size: 26px; font-weight: 700; }}
QLabel#H1 {{ font-size: 20px; font-weight: 600; }}
QLabel#H2 {{ font-size: 15px; font-weight: 600; }}
QLabel#CardIcon {{ font-size: 26px; }}
QLabel#Muted {{ color: {muted}; }}
QLabel#Metric {{ font-size: 22px; font-weight: 600; color: {accent}; }}
QLabel#Badge {{
    background: {surface2}; color: {muted}; border: 1px solid {border};
    border-radius: 4px; padding: 1px 6px;
}}

/* Navigation sidebar (grouped tree) */
QTreeWidget#Nav {{ background: {surface2}; border: none; border-right: 1px solid {border}; outline: 0; }}
QTreeWidget#Nav::item {{ padding: 8px 14px; margin: 1px 6px; border-radius: 6px; color: {muted}; }}
QTreeWidget#Nav::item:selected {{ background: {accent}; color: {accent_text}; }}
QTreeWidget#Nav::item:hover:!selected {{ background: {surface}; color: {text}; }}
QTreeWidget#Nav::branch {{ background: {surface2}; }}

/* Large action / project cards on the workspace */
QFrame#ActionCard {{ background: {surface}; border: 1px solid {border}; border-radius: 10px; min-height: 96px; }}
QFrame#ActionCard:hover {{ border: 1px solid {accent}; background: {surface2}; }}

/* Annotation workspace: compact toolbar, canvas, filmstrip */
QGraphicsView {{ background: {surface2}; border: 1px solid {border}; border-radius: 8px; }}
QListWidget::item:selected {{ background: {sel}; color: {text}; }}

QListWidget, QTableWidget, QTreeWidget, QPlainTextEdit, QTextEdit, QLineEdit, QComboBox {{
    background: {surface}; border: 1px solid {border}; border-radius: 6px;
    selection-background-color: {sel}; selection-color: {text};
}}
QLineEdit, QComboBox {{ padding: 5px 8px; }}
QHeaderView::section {{ background: {surface2}; color: {muted}; border: none;
    border-bottom: 1px solid {border}; padding: 6px; }}
QTableWidget {{ gridline-color: {border}; }}

QPushButton {{
    background: {surface2}; color: {text}; border: 1px solid {border};
    border-radius: 6px; padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {surface}; }}
QPushButton#Primary {{ background: {accent}; color: {accent_text}; border: none; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #4d8bff; }}
QPushButton:disabled {{ color: {muted}; border-color: {border}; background: {surface}; }}

QTabWidget::pane {{ border: 1px solid {border}; border-radius: 6px; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 6px 12px; }}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {accent}; }}

QProgressBar {{ background: {surface2}; border: 1px solid {border}; border-radius: 6px;
    text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}

QStatusBar {{ background: {surface2}; border-top: 1px solid {border}; color: {muted}; }}
QSplitter::handle {{ background: {border}; }}
QScrollBar:vertical {{ background: {bg}; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


class ThemeManager:
    THEMES = ("dark", "light")

    def __init__(self, app: QApplication, initial: str = "dark") -> None:
        self._app = app
        self._name = initial if initial in self.THEMES else "dark"
        self.apply(self._name)

    @property
    def name(self) -> str:
        return self._name

    def palette(self) -> dict:
        return _DARK if self._name == "dark" else _LIGHT

    def apply(self, name: str) -> None:
        self._name = name if name in self.THEMES else "dark"
        colors = _DARK if self._name == "dark" else _LIGHT
        self._app.setStyleSheet(_QSS.format(**colors))

    def toggle(self) -> str:
        self.apply("light" if self._name == "dark" else "dark")
        return self._name
