# AutoDataForge — Roadmap

A living view of what has shipped and what is likely next. The guiding principle is
constant: **one agentic pipeline** — Planner → Decision → Execution → Verification →
Memory → Export. New capabilities enter *above* the pipeline (as new dataset sources)
or *beside* it (as read-only intelligence surfaces) — never as a second pipeline, and
never by fabricating metrics.

## ✅ Shipped (v0.1)

- Goal-driven multi-agent orchestration (Planner, Decision, Execution, Memory agents
  + Task Orchestrator)
- Deterministic annotation pipeline (import → detect → segment → verify → export)
- Ultralytics YOLO detector/segmenter with a classical builtin fallback
- Provider-agnostic agent framework (Anthropic / OpenAI / Ollama / Echo)
- Deterministic, versioned Engineering Memory (no vector DB required)
- COCO / YOLO export
- Video import engine (probe → frame strategy → dedup → pipeline)
- MCP server exposing the platform to any MCP client
- PySide6 desktop application (dashboard, annotation, verification, intelligence,
  knowledge, operations workspaces)

## 🔜 Near term

- [ ] Confidence scores preserved in exported COCO/YOLO annotations
- [ ] Active-learning loop: route the most uncertain samples to human review first
- [ ] Additional export formats (Pascal VOC, CVAT, Label Studio JSON)
- [ ] Richer VLM verification (multi-model consensus)
- [ ] One-command Docker quick start with the GUI over VNC/web

## 🌅 Later

- [ ] Distributed execution across multiple GPUs / worker nodes
- [ ] Dataset versioning + lineage as first-class objects
- [ ] Web UI parity with the desktop app
- [ ] Plugin marketplace for detectors, segmenters, and verifiers
- [ ] Fine-tuning feedback loop (train → evaluate → recommend relabeling)

## Non-goals

- A second, parallel pipeline. Everything routes through the one orchestrated flow.
- Fabricated metrics. If a value cannot be measured, it is reported as *unavailable*.

Have an idea? Open a [feature request](https://github.com/Prasadslaxmi08/autodataforge/issues/new/choose).
