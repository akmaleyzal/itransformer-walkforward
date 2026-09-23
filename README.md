# iTransformer walk-forward

Walk-forward comparison of three forecasters of hourly Bitcoin log-returns, 24 hours ahead:
iTransformer (`itr`), Ridge regression (`rdg`) and a vanilla Transformer (`vtr`), each at
K ∈ {1, 4, 8, 12} variates, 15 origins and 5 seeds — 900 runs against a naive random-walk
reference. The design, its reasons and its references are in [`docs/DESIGN.md`](docs/DESIGN.md).

`itr` and `vtr` are the authors' `Model` classes from
[thuml/iTransformer @ c2426e68](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124)
(MIT), copied unchanged and checked by pinned sha256; `rdg` is scikit-learn `Ridge`.

## Layout

```
notebooks/btc_walkforward_3model.ipynb  the implementation, run on Kaggle
src/itransformer_btc/                   tested projection of the notebook
src/itransformer_btc/vendor/            upstream model files, byte-pinned
tests/                                  pytest suite, one file per concern
tools/                                  notebook sync and checks, report builder, upstream fetch
data/raw/                               Stage 1 Binance klines, sha256-pinned
docs/                                   design, window budget, notebook map
spot_klines_btc.py                      Stage 1 download (already run; not re-run)
```

## Run on Kaggle

1. Create a notebook from `notebooks/btc_walkforward_3model.ipynb`.
2. Accelerator **GPU T4 × 2**, Internet on.
3. Attach a dataset containing `data/raw/BTCUSDT_1h.parquet`. The notebook finds it and checks its sha256.
4. In the setup cell, set `WEEKLY_GPU_HOURS_REMAINING` to the quota Kaggle shows.
5. **Save Version → Save & Run All**.

The first session runs the validation-only pilot and prints a design digest; the grid does not start
until that digest is written into `DESIGN_FREEZE_SHA256` in the grid cell. Later sessions resume from
the previous version's output, attached as a dataset.

## Local checks

```powershell
uv sync --extra train --extra stats --extra plot --extra dev
.venv\Scripts\python.exe tools\build_notebook.py --check
.venv\Scripts\python.exe tools\run_tests.py
.venv\Scripts\python.exe tools\fetch_upstream.py
```

`run_tests.py` runs each test file in its own process. After editing the notebook, run
`tools\notebook_to_src.py` to refresh `src/` and the pinned code digest.

## License

MIT for this repository's code (see [`LICENSE`](LICENSE)); the upstream files keep their own MIT
license in `src/itransformer_btc/vendor/thuml_iTransformer/LICENSE`. Binance market data are
included for reproducibility of the study only.
