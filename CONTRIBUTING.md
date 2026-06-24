# Contributing

## Developer install

```bash
git clone https://github.com/ajmayer95/net3.git
cd net3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[gui,dev]'
pytest -q
```

Edits to `src/net3/` take effect immediately (no reinstall needed).
`pytest -q` reports 32 passed.

## Layout

- `src/net3/` — package source
  - `pipeline.py` — top-level `vectorize()` function (mask → graph)
  - `image.py`, `contours.py`, `triangulate.py`, `graph.py`,
    `rediscretize.py` — pipeline stages
  - `cli.py` — `net3 vectorize` / `net3 edit` entry points
  - `C_net_functions.pyx` — Cython extension (triangle pruning + adjacency)
  - `edit/` — interactive editor
    - `core.py` — GUI-free `GraphEditor` backend
    - `app.py` — napari frontend (loaded only when `[gui]` extra is installed)
- `tests/` — pytest test suite
- `examples/` — runnable example scripts

## Adding tests

Backend tests live in `tests/test_edit_core.py` and `tests/test_pipeline.py`.
The backend (`GraphEditor`, `vectorize`) is testable without napari —
new tests should follow that pattern unless they're explicitly
exercising the GUI.

## Filing an issue / PR

Open at <https://github.com/ajmayer95/net3/issues>. Pull requests
welcome; please include a test for any behavior change.
