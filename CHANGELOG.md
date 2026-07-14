# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Rebranded the project to **AutoDataForge** and repositioned it as an agentic AI
  platform for intelligent dataset engineering.
- Renamed the distribution to `autodataforge` with `autodataforge` / `adf` console
  entry points. The Python import package remains `vds`.
- Verifier default policy is now binary: annotations with confidence ≥ 0.50 are
  approved, below 0.50 are rejected (the needs-review band is still available via
  custom thresholds).
- Repository restructured for open-source release: added standard community health
  files, CI, and a professional README; removed development-only artifacts.

### Fixed
- `generate_report` no longer crashes on review-/export-only plans that have no
  pipeline `ExecutionReport` (guarded the `None` case in `report_markdown`).

## [0.1.0] — 2026-07-14

### Added
- Goal-driven multi-agent layer: **Planner**, **Decision**, **Execution**, and
  **Memory** agents coordinated by a deterministic **Task Orchestrator**.
- **MCP server** exposing the platform as Model Context Protocol tools.
- Deterministic annotation pipeline: import → plan → detect → segment → verify →
  export, with COCO / YOLO output.
- Ultralytics YOLO detector/segmenter with a classical builtin fallback.
- Provider-agnostic agent framework (Anthropic / OpenAI / Ollama / Echo).
- Deterministic, versioned **Engineering Memory** (no vector DB required).
- PySide6 desktop application with dashboard, annotation, verification,
  intelligence, knowledge, and operations workspaces.

[Unreleased]: https://github.com/Prasadslaxmi08/autodataforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Prasadslaxmi08/autodataforge/releases/tag/v0.1.0
