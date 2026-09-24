# Study design

This document states what the study measures, how, and why each choice is defensible. It is declared before the test period is opened; nothing here was chosen after seeing test results. A change made after the design is frozen gets a numbered entry at the end of this file, with its reason.

## 1. Questions

The study compares three forecasters of hourly Bitcoin log-returns, 24 hours ahead, under one walk-forward protocol:

| Code | Model | Source |
|---|---|---|
| `itr` | iTransformer | `model/iTransformer.py`, `thuml/iTransformer` @ `c2426e68` (MIT), unchanged |
| `rdg` | Ridge regression | scikit-learn `Ridge` |
| `vtr` | vanilla Transformer | `model/Transformer.py`, same repository and commit, unchanged |

| Code | Question |
|---|---|
| **C1** | Does `itr` beat `rdg` on the same information, at each number of variates K? |
| **C2** | How does `itr` perform against `vtr`, overall and at each K? This compares whole architectures; it does not isolate inverted tokens. |
| **RQ1** | Does the benefit of added variates track nominal K or the effective dimensionality K_eff? |
| **RQ2** | Does the K=1-versus-K=8 gap of `itr` narrow with time since training? Descriptive, reported with its minimum detectable effect. |

The reference forecast is the naive random walk, `r̂ = 0`, which needs no training. The study is an accuracy comparison; it makes no claim about a tradable strategy and computes no economic metric.

## 2. Data

- Binance BTCUSDT spot, 1-hour klines, 2018-01-01T00:00Z to 2026-08-01T00:00Z (end exclusive), downloaded once by `spot_klines_btc.py` from the public REST API.
- Stage 1 artifacts in `data/raw/` are immutable and pinned by sha256: `BTCUSDT_1h.parquet` `8270a84b…`, `BTCUSDT_1h_gaps.csv` `cfab4cf4…`, `BTCUSDT_1h_raw.jsonl` `30721a66…`, `BTCUSDT_1h_report.json` (full digests inside it and in `config.py`). The notebook never re-downloads: a new download is a new data vintage.
- 75,216 expected and 75,094 actual bars; 122 missing bars in 27 gap blocks, the largest 33 hours; 3 zero-volume, zero-trade bars.
- All eleven meaningful kline columns are kept.
- All timestamps are UTC epoch integers, compared as integers.

**No imputation.** A missing bar is an hour in which no price was recorded in this artifact; its cause is not verified. Forward-filling or interpolating it would invent a price, so the series is split instead:

- A segment breaks at every missing bar and at every zero-volume or `high == low` bar.
- Log-returns are computed within a segment; each segment's first bar is dropped.
- A window `[s, s+L+H)` is valid only if its last timestamp is exactly `L+H−1` hours after its first. Validity is checked on timestamps, never on positions.
- Windows are counted segment-wise, `Σ max(0, nᵢ − L − H + 1)`; the per-origin budget in `docs/ORIGIN_WINDOW_BUDGET.md` is asserted by exact equality.
- Extreme returns are kept: no winsorizing, clipping or dropping.

Variance ratio (Lo and MacKinlay, 1988), Hurst (R/S) and ADF are reported on the full sample and on each origin's training sub-block, as description. They do not establish market efficiency.

## 3. Variates and the K ladder

Every variate is a function of one bar (the return also reads the previous close). No rolling window is computed on any feature, which is what makes an embargo unnecessary (section 5).

| Family | Variates |
|---|---|
| F1 price | `r = log(C/C₋₁)`, `upper_shadow = log(H/max(O,C))`, `lower_shadow = log(min(O,C)/L)` |
| F2 volatility | log Parkinson, log Garman–Klass, Rogers–Satchell as `log(RS + 1e-9)` (RS is zero on shadowless bars) |
| F3 intensity | `log_quote_volume`, `log_trade_count`, `log_mean_trade_size` |
| F4 order flow | `taker_buy_ratio = taker_buy_base_volume / volume`; `signed_flow = (2·taker_buy_ratio − 1)·log_quote_volume` |
| F5 location | `vwap_location = (VWAP − C)/(H − L)`, `VWAP = quote_volume/volume` |

| K | Adds | Role |
|---|---|---|
| 1 | `r` | univariate control |
| 4 | `upper_shadow`, `lower_shadow`, `log_quote_volume` | |
| 8 | `log_trade_count`, `taker_buy_ratio`, `signed_flow`, `vwap_location` | |
| 12 | Parkinson, Garman–Klass, Rogers–Satchell, `log_mean_trade_size` | deliberately redundant control |

- The ladder is fixed before any training; no rung, lookback or configuration is added after results are seen.
- `r` is always channel 0 and is the only forecast target.
- K_eff is the participation ratio `(Σλ)²/Σλ²` of a rung's correlation matrix, computed per origin on that origin's training sub-block only. Also reported: PR on window-normalised features, a lookback-aware correlation PR as a fraction of `K·L`, and stable rank.
- The pre-first-origin PR gate uses 2018-01 to 2020-01 only. PR(K=8) below 5.0 is disclosed, not re-cut. The measured value is printed by the notebook.
- PR describes coordinates, not information; no causal claim is made about it.

## 4. Models

All three models see the same windows, the same scaler and the same 11,500 training windows per (origin, K); train with MSE on the target channel only; use seeds 42–46; and have no per-rung tuning.

**`itr`**: `Linear(96→128)` embeds each variate's whole lookback as one token; attention runs over the K variate tokens; `Linear(128→24)` is read at the target channel. `use_norm=True` (per-window instance normalisation with a detached mean) is part of the architecture, not a tuning knob.

**`rdg`**: scikit-learn `Ridge(fit_intercept=True, solver="cholesky")` in float64, from the flattened `K·96` lookback to 24 outputs. α is chosen on validation MSE from the eight values 1e-1, 1e0, …, 1e6 (`RIDGE_ALPHAS` in `baselines.py`), the only selected hyperparameter in the study. The solve is closed-form on a fixed sample, so the five seeds give identical predictions; they are kept for a symmetric run manifest and reported as seed std = 0.

**`vtr`**: Conv1d token embedding with sinusoidal positional encoding (no calendar marks); 2 encoder and 1 decoder layers; the decoder reads the last 48 lookback hours plus 24 zero placeholders under a causal mask and emits all 24 hours in one pass; `Linear(128→K)` read at the target channel; no instance normalisation.

| Setting | `itr` | `vtr` |
|---|---|---|
| `d_model` / FFN / heads / dropout | 128 / 256 / 8 / 0.1 | 128 / 256 / 8 / 0.1 |
| Layers | 2 encoder | 2 encoder + 1 decoder, `label_len` 48 |
| Activation | GELU | GELU |
| Parameters (K = 1/4/8/12) | 280,728 at every K | 466,177 / 468,868 / 472,456 / 476,044 |
| Schedule | Adam, lr 1e-4 halved every 4 epochs, batch 32, ≤ 30 epochs, patience 5 on validation MSE | same |
| Precision | float32, no mixed precision | float32, no mixed precision |

- Hyperparameters follow Liu et al. (2024) except `d_model` 512 → 128 for the sample size, identical at every rung.
- `vtr` settings (2 + 1 layers, GELU, `label_len` 48) are the defaults of the official `run.py` in the iTransformer code base and the related long-horizon forecasting code bases.
- `vtr` departs from Vaswani et al. (2017) in one-pass decoding from a known-history start token, 2 + 1 layers at `d_model` 128 (paper: 6 + 6 at 512), GELU (paper: ReLU), this study's learning-rate schedule instead of warm-up, and no label smoothing. One-pass decoding matches the other two models; iterated decoding shrinks forecast variance (Zeng et al., 2023).
- `vtr` has more parameters than `itr`. No parameter count licenses "same effective capacity"; Table 3 prints parameters and each model's count of runs that reached the epoch cap.
- At K = 1, attention over one token has weight 1, but the value/output projections and the residual remain: a designed control. K = 1 and K = 8 differ both in information and in whether attention is active; no contrast here isolates attention.
- `use_norm` cancels any per-channel affine scaler, so the scaler matters for learning only in `rdg` and `vtr`. The invariance `MSE(c·x)/c² = MSE(x)` is tested before the grid.
- StandardScaler is used rather than RobustScaler (inflates outliers under fat tails) or MinMaxScaler (out of range on new regimes).

**Provenance.** `src/itransformer_btc/vendor/thuml_iTransformer/` holds the upstream files at commit `c2426e68ca13f74aaec08045c5c724d8ad328124`. They differ from the published files only in whitespace (one final newline, whitespace-only lines emptied), which is the form a notebook `%%writefile` cell writes. `layers/SelfAttention_Family.py` keeps only `FullAttention` and `AttentionLayer`; its `reformer_pytorch`/`einops` imports serve classes this study does not use. Pinned sha256 values are checked by the tests and by the notebook, and `tools/fetch_upstream.py` re-downloads the published files and compares them.

## 5. Walk-forward protocol

- 15 origins at 5-month spacing: 2020-01, 2020-06, 2020-11, 2021-04, 2021-09, 2022-02, 2022-07, 2022-12, 2023-05, 2023-10, 2024-03, 2024-08, 2025-01, 2025-06, 2025-11.
- Per origin, a fixed 24-month rolling window: a 21-month training sub-block and a 3-month validation sub-block, then six 30-day test blocks with no retraining.
- Lookback L = 96 hours, horizon H = 24 hours.
- Each run trains on exactly 11,500 windows drawn without replacement (seed 1729) from the training time index. The scaler is fitted first, on the purged training bars.
- Targets are purged by H steps at both the train→validation and the train→test boundary. Validation and test inputs may reach back into earlier periods; that is information a forecaster legitimately has.
- Forecast origin = opening time of the first target bar. Test windows are assigned to blocks by issuance time.
- 900 runs: 3 models × 15 origins × 4 K × 5 seeds.

Why each element is defensible:

| Element | Reason | Grounding |
|---|---|---|
| Many origins, not one split | One split measures one regime; its error has a sample size of one origin. | Tashman (2000); Bergmeir and Benítez (2012); Hyndman and Athanasopoulos (2021) |
| Chronological order, no K-fold | RQ2's variable is time since training, which reordering destroys. The theorem that validates K-fold CV for time series assumes a correctly specified stationary autoregression with martingale-difference errors; errors of a 24-step forecast are MA(23), and a multivariate model is outside the theorem's model class. On non-stationary real series CV underestimates error. | Bergmeir, Hyndman and Koo (2018); Cerqueira, Torgo and Mozetič (2020) |
| Rolling, not expanding | With an expanding window, model age and training-set size move together. Rolling windows "level the playing field in a multiperiod comparison". | Tashman (2000); Giacomini and White (2006); Pesaran and Timmermann (2007); Rossi (2013) |
| 24-month window | Fixed, not estimated; enough windows for 12 × 96 inputs. Window length is a free parameter with consequences, stated as a limitation. | Inoue, Jin and Rossi (2017) |
| 21/3 train–validation split | Early stopping and α must be selected before the origin; the split is declared before the test period is opened. | Hansen and Timmermann (2015); Arnott, Harvey and Markowitz (2019) |
| H-step purge at both boundaries | A training target reaching past a boundary carries later observations into training, including the split that governs model selection. | López de Prado (2018); Kaufman et al. (2012) |
| No embargo | An embargo guards against a test bar influencing a training feature; with per-bar features that channel does not exist. This must be re-derived if a multi-bar feature is ever added. | López de Prado (2018); Kaufman et al. (2012) |
| Frozen weights across the test span | The inputs still roll forward at every issuance; only the weights stay fixed, and that is what RQ2 measures. Each origin is a full refit, i.e. recalibration every five months. | Tashman (2000) |
| 5-month spacing | Block b of origin i starts in month `m₀ + s·i + (b−1) mod 12`. With s = 6 each block index visits two calendar months, so a block trend would be indistinguishable from a month-of-year effect, which exists in Bitcoin. Only s coprime to 12 decouples them. This is the study's own choice. | Baur et al. (2019) for the premise |
| Clustered by origin | Consecutive origins share 79.2% of training data (58.3/37.5/16.7% at strides 2–4); only stride-5 triples are training-disjoint. About four effectively independent training sets. | Cameron, Gelbach and Miller (2008); MacKinnon, Nielsen and Webb (2023) |
| CPCV rejected | It orders blocks non-chronologically, which leaves time since training undefined, and assumes block-to-block stability. The comparison that favours CPCV used a single-path, unpurged walk-forward. | Arian, Norouzi Mobarekeh and Seco (2024); López de Prado (2018) |
| Test opened once | The pilot uses validation only; the grid refuses to run until the design digest is frozen. | Arnott, Harvey and Markowitz (2019) |

Leakage checks (F = fatal, each backed by a test or a notebook assertion):

- **F** Returns per segment before scaling; windows validated by timestamp; parquet rows equal the report's actual bar count and all 27 gap blocks are present.
- **F** Scaler per origin on the training sub-block only; naive-RW mapped to `ŷ_z = −μ_g/σ_g`, with `μ_g` and `μ_g/σ_g` logged per origin.
- **F** Last training target before validation start; last validation target before test start.
- **F** Loss on the target channel only; `use_norm` active and its invariance test passing.
- **F** Every K_eff computed on a training-only span.
- Every comparison is scored on the exact timestamps all compared runs share.
- Each run's training selection (count, seed, timestamp digest) is written to `meta/*.json`; raw predictions are saved for every run.

## 6. Metrics

- Naive-RW is `r̂ = 0` in raw log-return space, mapped to `ŷ_z = −μ_g/σ_g` in scaler space.
- `RelMSE(b) = MSE_model(b) / MSE_naive(b)` on the same block; `R²_oos = 1 − RelMSE`. A ratio against the naive forecast removes each period's difficulty from the comparison (Tashman, 2000).
- Aggregation: squared error per step within a seed → mean over seeds → RelMSE per block → equal block weight → equal origin weight. Every ratio is formed from seed-averaged MSEs, never by averaging per-seed ratios.
- Comparisons use the common calendar: target hours present for every run compared, full horizon. The calendar hash is stored with each paired contrast.
- Cross-origin comparisons use RelMSE or `R²_oos`, never scaler-space MSE, because each origin has its own `σ_g`.
- MSE is reported in scaler space and RMSE in raw log-return units, with `σ_g` stated. Signs and sums use raw returns `r = zσ_g + μ_g`.
- RQ2's dependent variable is `A(i,b) = [MSE_K1 − MSE_K8] / MSE_K1` on the `itr` ladder.

**Directional accuracy.**

- DA-1h is the sign of the step-1 return at every issuance hour. DA-24h is the sign of the step-24 return. DA-cum is the sign of `σ_g·Σz + 24μ_g`. The last two use one issuance per day, at 00:00 UTC.
- Zero actual returns are excluded, and a zero prediction counts as wrong. Naive-RW has no DA.
- Baseline per origin and variant is `max(p_up, 1 − p_up)` on the common calendar. It uses test-period frequencies, so it bounds any constant-sign guess from above, which makes it conservative.
- `ΔDA = DA − baseline` is computed per origin (seed-averaged first) and reported as mean ± SE across the 15 origins, with the count of origins where ΔDA > 0.
- A Pesaran–Timmermann (1992) test per origin is reported as a diagnostic count only.

## 7. Inference

All inference is exploratory. Origins share training data, and a multiplicity correction does not repair unmodelled dependence. Every bootstrap (Romano–Wolf, the Model Confidence Set, the wild cluster bootstrap) uses B = 9,999 draws with seed 42, so the smallest attainable p is 1/10,000.

- **Diebold–Mariano** (1995) per (origin, block): T ≈ 720, h = 24, rectangular long-run variance truncated at lag h − 1 (Bartlett fallback if the variance is not positive, reported). The Harvey–Leybourne–Newbold (1997) correction is applied against t(T−1), and T is printed beside every p. `d_t` is never concatenated across origins.
- **Clark–West** (2007) is run only against naive-RW, the one true nesting, as a diagnostic. The K ladder is not assumed nested, and a positive CW can sit beside a negative `R²_oos`.
- **Romano–Wolf** (2005) stepdown is applied over all pairs and within the families `ladder`, `vs-naive` and `cross-model` (`itr` vs `rdg`, `itr` vs `vtr`, per K).
- **Model Confidence Set** (Hansen, Lunde and Nason, 2011) at 90% and 75% over 13 models (3 × 4 K plus naive-RW).
- **RQ1:** `MSE(i,b,K) = γ_ib + f(K) + ε` with (origin × block) fixed effects clustered by origin. K and K_eff are compared by a non-nested J-test. The 8→12 rung is tested for equivalence with TOST at margin `0.25 × ΔMSE₄→₈`.
- **RQ2:** `A(i,b) = αᵢ + β₁b + ε` with origin fixed effects, clustered by origin, G = 15. The test uses a restricted wild cluster bootstrap of the cluster-robust t with Rademacher and Webb weights (the more conservative is the headline), p = `(1 + count)/(1 + B)`, reference t(G−1), one-sided β₁ < 0 at α = 0.05. The post-analysis minimum detectable effect is printed beside β₁, together with a block-coverage covariate and all five stride-5 triples with their spread.
- Dispersion follows the aggregation level. Per (origin, block): mean ± std over seeds, with n. Across origins: mean ± SE across origins, with seed std as a separate diagnostic.
- A paired contrast between two arms that share no origin raises an error; G = 0 is a bug, not a null.

## 8. Execution

- The notebook `notebooks/btc_walkforward_3model.ipynb` is the implementation; `src/itransformer_btc/` is its tested projection (`tools/notebook_to_src.py`).
- Kaggle, 2 × T4 GPUs. One worker thread per device runs whole runs in parallel (no DataParallel). Each split is loaded onto the device once and batched by index slicing.
- Runs execute in the order `itr`, `rdg`, `vtr`, keyed `{model}_o{origin:02d}_K{K:02d}_H024_s{seed}`.
- One monotonic session deadline covers pilot and grid. Checkpoints are written every epoch, and a partial epoch replays from its last minibatch boundary.
- A run counts as complete only when the code digest, input digest, configuration, schedule, schema, and prediction and weight hashes all match; `meta` is written last.
- **Pilot:** origin 1, all three models × four K × one seed, trained on the training sub-block and scored on validation only. It checks that the pipeline runs and measures wall time per model. No configuration is selected from it.
- **Design freeze:** `design_digest()` hashes the code digest together with the 900 run IDs. The grid step refuses to run until `DESIGN_FREEZE_SHA256` in the notebook equals the digest printed by the pilot. Any code change after the pilot changes the digest and requires a new pilot.
- Estimators run only on a complete panel; partial evaluation is never a fallback.

## 9. Disclosures

- Nothing was externally pre-registered; the design is declared here before the test period is opened. The MDE and the equivalence margin are applied post-analysis.
- Inference is diagnostic: 79.2% overlap between consecutive origins, about four effectively independent training sets.
- Clark–West is not applied to the ladder.
- K = 1 attention degeneracy; K = 12 redundancy; the PR gate result is disclosed as measured.
- The `use_norm` confound on F2: the volatility family contributes shape, not level.
- No imputation, and no claim about the cause of a gap. Test windows near gaps are excluded, and that exclusion depends on the future. The data are revised, not real-time. Single venue and single pair.
- `vtr`'s departures from Vaswani et al. (2017) and its larger parameter count; the at-cap run counts for every model.
- `rdg` is deterministic (seed std = 0).
- The DA baseline uses ex-post test frequencies.
- RQ2 is descriptive and reported with its MDE.
- Results are bound to the models, target, horizon, preprocessing, sample and aggregation tested.

## References

- Arian, H., Norouzi Mobarekeh, D., & Seco, L. (2024). Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment. *Knowledge-Based Systems*. https://doi.org/10.1016/j.knosys.2024.112477
- Arnott, R. D., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the era of machine learning. *The Journal of Financial Data Science*. https://doi.org/10.3905/jfds.2019.1.064
- Baur, D. G., Cahill, D., Godfrey, K., & Liu, Z. (2019). Bitcoin time-of-day, day-of-week and month-of-year effects in returns and trading volume. *Finance Research Letters*. https://doi.org/10.1016/j.frl.2019.04.023
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*. https://doi.org/10.1016/j.ins.2011.12.028
- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*. https://doi.org/10.1016/j.csda.2017.11.003
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *The Review of Economics and Statistics*. https://doi.org/10.1162/rest.90.3.414
- Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*. https://doi.org/10.1007/s10994-020-05910-7
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2006.05.023
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*. https://doi.org/10.1080/07350015.1995.10524599
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*. https://doi.org/10.1111/j.1468-0262.2006.00718.x
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*. https://doi.org/10.3982/ECTA5771
- Hansen, P. R., & Timmermann, A. (2015). Equivalence between out-of-sample forecast comparisons and Wald statistics. *Econometrica*. https://doi.org/10.3982/ECTA10581
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*. https://doi.org/10.1016/S0169-2070(96)00719-4
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*. https://doi.org/10.1080/00401706.1970.10488634
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Inoue, A., Jin, L., & Rossi, B. (2017). Rolling window selection for out-of-sample forecasting with time-varying parameters. *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2016.03.006
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*. https://doi.org/10.1145/2382577.2382579
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR. https://doi.org/10.48550/arXiv.1412.6980
- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted Transformers are effective for time series forecasting. ICLR. arXiv:2310.06625. Code: https://github.com/thuml/iTransformer
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*. https://doi.org/10.1093/rfs/1.1.41
- López de Prado, M. (2018). *Advances in financial machine learning*. Wiley. ISBN 978-1-119-48208-6
- MacKinnon, J. G., Nielsen, M. Ø., & Webb, M. D. (2023). Cluster-robust inference: A guide to empirical practice. *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2022.04.001
- Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*. https://doi.org/10.1080/07350015.1992.10509922
- Pesaran, M. H., & Timmermann, A. (2007). Selection of estimation window in the presence of breaks. *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2006.03.010
- Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*. https://doi.org/10.1111/j.1468-0262.2005.00615.x
- Rossi, B. (2013). Advances in forecasting under instability. In *Handbook of Economic Forecasting* (Vol. 2). https://doi.org/10.1016/B978-0-444-62731-5.00021-X
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*. https://doi.org/10.1016/S0169-2070(00)00065-0
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. NeurIPS. arXiv:1706.03762
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers effective for time series forecasting? AAAI. https://doi.org/10.1609/aaai.v37i9.26317

## Design freeze

Frozen on 2026-09-24, after the validation-only pilot and before any test block was scored.

| Item | Value |
|---|---|
| Design digest (code digest and the 900 run IDs) | `9f2435aefac51199e6906f68ac7c3952d9bcc1638fe39873cb04f01c0b1ad7db` |
| Code digest (`code_sha256`, package and upstream copies) | `5a23e2725b00005c0870e7b3e4fc69f94125277e9492ff2d660db2208c207116` |
| Input digest (`BTCUSDT_1h.parquet`) | `8270a84b07c2923bc885782a8ba4e1898133d18ee3b260f157fcee3fd6923b4e` |
| Upstream | `thuml/iTransformer` @ `c2426e68ca13f74aaec08045c5c724d8ad328124` |
| Pilot environment | Kaggle, 2 × Tesla T4 (sm_75), torch 2.10.0+cu128, polars 1.35.2, numpy 2.0.2 |
| Pilot cost, mean wall time per run | `itr` 40.2 s, `rdg` 2.5 s, `vtr` 49.0 s |
| Projected grid cost | `itr` 3.35, `rdg` 0.21, `vtr` 4.08 GPU-hours; about 3.8 h of wall-clock time on two T4s |

- The pilot fitted each model at every K on origin 1's training sub-block with seed 42 and scored the validation sub-block only; no test prediction exists from it. It checks that every validation loss is finite and measures cost. Its validation errors are not a basis for any choice: no setting changed after it, and no hyperparameter search precedes the grid; the ridge α chosen on validation is the only selection in the study.
- The notebook's grid step pins this digest as `DESIGN_FREEZE_SHA256` and refuses to train when the code produces another; `tests/test_notebook.py` fails when the committed code no longer produces it.
- No notebook cell is edited until the grid is complete. Between sessions only the operator fields of the setup cell change (`WEEKLY_GPU_HOURS_REMAINING`, `SESSION_ALREADY_USED_H`). An unavoidable fix makes a new code vintage: re-run the pilot, freeze again, and record it below.
- All 5 seeds, 15 origins and 6 test blocks are reported. No run is dropped or rerun silently, and every session's log is kept.

## Changes after the design freeze

None yet.
