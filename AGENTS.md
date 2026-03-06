# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python codebase organized at the repo root.
- `Equipment.py` defines the core VPP device hierarchy (`Equipment`, `PV`, `WIND`, `DG`, `ESS`, `DL`, `TCR`) plus shared capacity and bidding helpers.
- `NTCRVppFull.py` contains the higher-level device manager and VPP optimization / dispatch logic built around NumPy, pandas, CPLEX `docplex`, and Flask context hooks.
- There is no `tests/` directory yet. Add new tests under `tests/` instead of mixing test code into production modules.

## Build, Test, and Development Commands
Set up the environment manually because the repository does not yet include packaging metadata.
- `python -m venv .venv && source .venv/bin/activate` — create a local virtual environment.
- `pip install numpy pandas flask docplex` — install the runtime dependencies used by the current modules.
- `python -m py_compile Equipment.py NTCRVppFull.py` — run a quick syntax check before committing.
- `pytest -q` — run automated tests after a `tests/` suite is added.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8 where practical. Keep functions and variables in `snake_case` and classes in `PascalCase`. Preserve the existing uppercase device-type identifiers (`PV`, `WIND`, `DG`, `ESS`, `DL`, `TCR`) because they are used directly in data structures and bidding logic. Prefer type hints for new public methods, keep docstrings short, and make units explicit in kW / kWh-related code.

## Testing Guidelines
Use `pytest` for new automated tests. Name files `tests/test_<module>.py` and keep fixtures small and deterministic. Prioritize coverage around device aggregation, status filtering, bid-ratio limits, and optimization guardrails. For every bug fix, add at least one regression test that reproduces the failing device or owner scenario.

## Commit & Pull Request Guidelines
The current Git history uses short, descriptive Chinese commit summaries (for example, `初始提交：虚拟电厂(VPP)设备模型与控制器`). Keep commits focused and single-purpose. Recommended styles include `fix: 修正ESS功率边界计算` and `refactor: 拆分设备聚合逻辑`. Pull requests should explain the business scenario, impacted modules, validation commands, and any solver or import-path assumptions.

## Security & Configuration Tips
Do not commit solver licenses, secrets, or real customer / owner identifiers. If you introduce package structure or external config files, document import-path changes clearly because `NTCRVppFull.py` currently relies on relative imports.
