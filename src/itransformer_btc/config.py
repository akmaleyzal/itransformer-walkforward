"""Design constants, the walk-forward origin grid, and where each algorithm came from.

Every number here is fixed before any model runs, so no magic number is buried
in pipeline code. The origin grid is derived from the constants rather than
written out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

# -- data window ---------------------------------------------------------------

DATA_START: Final = datetime(2018, 1, 1, tzinfo=timezone.utc)
DATA_END: Final = datetime(2026, 8, 1, tzinfo=timezone.utc)  # EXCLUSIVE

BARS_EXPECTED: Final = 75_216
BARS_ACTUAL: Final = 75_094
MISSING_BARS: Final = 122
GAP_BLOCKS: Final = 27

#: sha256 of ``data/raw/BTCUSDT_1h.parquet``, the one input the study reads.
INPUT_SHA256: Final = "8270a84b07c2923bc885782a8ba4e1898133d18ee3b260f157fcee3fd6923b4e"

# -- window geometry -----------------------------------------------------------

SEQ_LEN: Final = 96   # L: four days of lookback
PRED_LEN: Final = 24  # H: one day ahead

#: A window spans ``L + H`` bars, so one break removes ``L + H - 1`` start positions.
WINDOW_SPAN: Final = SEQ_LEN + PRED_LEN         # 120
STARTS_LOST_PER_BREAK: Final = WINDOW_SPAN - 1  # 119

# -- walk-forward protocol -----------------------------------------------------

TRAIN_MONTHS: Final = 24      # fixed rolling window, never expanding
VAL_MONTHS: Final = 3         # final 3 months of the training window
TRAIN_SUB_MONTHS: Final = TRAIN_MONTHS - VAL_MONTHS  # 21: where the scaler is fitted
TEST_BLOCKS: Final = 6
BLOCK_DAYS: Final = 30
BLOCK_HOURS: Final = BLOCK_DAYS * 24  # 720 forecast origins per block

#: Five months between origins. A spacing coprime to 12 keeps the test block
#: index from tracking the calendar month; 5 gives the most origins among those.
ORIGIN_SPACING_MONTHS: Final = 5

FIRST_ORIGIN: Final = datetime(2020, 1, 1, tzinfo=timezone.utc)

# -- ladder, seeds and the training sample ------------------------------------------

K_LADDER: Final = (1, 4, 8, 12)
SEEDS: Final = (42, 43, 44, 45, 46)

#: Training windows drawn per run, identical for all three models at a cell.
TRAIN_WINDOW_LIMIT: Final = 11_500
SELECTION_SEED: Final = 1729


def add_months(when: datetime, months: int) -> datetime:
    """Shift a first-of-month datetime by whole calendar months.

    Raises:
        ValueError: If ``when`` is not on the first of a month; clamping a day
            would move a split silently.
    """
    if when.day != 1:
        raise ValueError(
            f"add_months is only used on month boundaries in this study; got "
            f"day={when.day}. Clamping rules would silently move a split."
        )
    total = when.month - 1 + months
    return when.replace(year=when.year + total // 12, month=total % 12 + 1)


@dataclass(frozen=True, slots=True)
class Origin:
    """One walk-forward origin and every boundary derived from it.

    Boundaries are half-open ``[start, end)``. The origin ends validation and
    starts testing: a forecaster at ``o`` has seen everything before ``o``.
    """

    index: int
    origin: datetime

    @property
    def train_start(self) -> datetime:
        """Start of the 24-month rolling training window."""
        return add_months(self.origin, -TRAIN_MONTHS)

    @property
    def train_sub_end(self) -> datetime:
        """End of the 21-month training sub-block, which is also ``val_start``."""
        return add_months(self.origin, -VAL_MONTHS)

    @property
    def val_start(self) -> datetime:
        return self.train_sub_end

    @property
    def val_end(self) -> datetime:
        return self.origin

    @property
    def test_start(self) -> datetime:
        return self.origin

    @property
    def test_end(self) -> datetime:
        return self.origin + timedelta(days=BLOCK_DAYS * TEST_BLOCKS)

    def block(self, b: int) -> tuple[datetime, datetime]:
        """Half-open bounds of test block ``b``, one-indexed."""
        if not 1 <= b <= TEST_BLOCKS:
            raise ValueError(f"block index must be in 1..{TEST_BLOCKS}, got {b}")
        start = self.origin + timedelta(days=BLOCK_DAYS * (b - 1))
        return start, start + timedelta(days=BLOCK_DAYS)

    def blocks(self) -> list[tuple[int, datetime, datetime]]:
        """Every test block as ``(label, start, end)``, label one-indexed."""
        return [(b, *self.block(b)) for b in range(1, TEST_BLOCKS + 1)]

    @property
    def label(self) -> str:
        """``YYYY-MM``, the form used in every table and figure."""
        return self.origin.strftime("%Y-%m")


#: What :func:`itransformer_btc.splits.build_origin_tensors` accepts.
OriginLike = Origin


def origin_grid(
    first: datetime = FIRST_ORIGIN,
    spacing_months: int = ORIGIN_SPACING_MONTHS,
    data_start: datetime = DATA_START,
    data_end: datetime = DATA_END,
) -> list[Origin]:
    """Every origin whose 24-month training window and six test blocks fit the data.

    Under the constants above this yields 15 origins, 2020-01 to 2025-11.

    Raises:
        ValueError: If the first origin would need data from before the window.
    """
    grid: list[Origin] = []
    candidate = first
    while True:
        origin = Origin(index=len(grid) + 1, origin=candidate)
        if origin.train_start < data_start:
            raise ValueError(
                f"origin {origin.label} needs training data from "
                f"{origin.train_start.date()}, before the data window opens at "
                f"{data_start.date()}"
            )
        if origin.test_end > data_end:
            break
        grid.append(origin)
        candidate = add_months(candidate, spacing_months)
    return grid


#: Materialised once; import this rather than rebuilding the grid.
ORIGINS: Final = origin_grid()


# -- provenance of every algorithm the study runs -------------------------------


@dataclass(frozen=True, slots=True)
class Upstream:
    """Where one algorithm in this package came from, and what was changed.

    Attributes:
        component: The names in this package the row accounts for.
        module: The ``src/itransformer_btc`` file they live in.
        status: ``copied`` (the authors' code, unchanged), ``library`` (imported
            and called) or ``own`` (written here from the published description).
        reference: IEEE-style citation.
        repo: Official code, empty when there is none.
        licence: Upstream licence, empty when there is no upstream code.
        accessed: ISO date the repository was last opened.
        adapted: Every deliberate departure, or how the code is called.
        verified: True when the repository at the stated revision was checked.
    """

    component: str
    module: str
    status: str
    reference: str
    repo: str = ""
    licence: str = ""
    accessed: str = ""
    adapted: str = ""
    verified: bool = False


#: Printed by the notebook's provenance step and bound to the module docstrings
#: by ``tests/test_provenance.py``.
SOURCE_PROVENANCE: Final[tuple[Upstream, ...]] = (
    Upstream(
        component="ITransformerForecaster (Model in model/iTransformer.py)",
        module="model.py",
        status="copied",
        reference=(
            "Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, "
            '"iTransformer: Inverted transformers are effective for time series '
            'forecasting," in Proc. 12th Int. Conf. Learn. Represent. (ICLR), '
            "2024. arXiv:2310.06625."
        ),
        repo="https://github.com/thuml/iTransformer",
        licence="MIT",
        accessed="2026-09-23",
        adapted=(
            "Copied unchanged at commit c2426e68ca13f74aaec08045c5c724d8ad328124. "
            "The adapter passes x_mark=None, reads the target channel, and uses "
            "d_model 128 and d_ff 256 for the sample size; use_norm stays on."
        ),
        verified=True,
    ),
    Upstream(
        component="VanillaForecaster (Model in model/Transformer.py)",
        module="model.py",
        status="copied",
        reference=(
            "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. "
            'Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in '
            "Adv. Neural Inf. Process. Syst. 30 (NeurIPS), 2017; implementation "
            "from the iTransformer repository (Liu et al., 2024)."
        ),
        repo="https://github.com/thuml/iTransformer",
        licence="MIT",
        accessed="2026-09-23",
        adapted=(
            "Copied unchanged at the same commit, with the official code's "
            "defaults: 2 encoder and 1 decoder layer, gelu, label_len 48, "
            "x_mark=None. The decoder reads the last 48 lookback hours plus 24 "
            "zero placeholders and decodes in one pass; the target channel is "
            "read from the all-channel head."
        ),
        verified=True,
    ),
    Upstream(
        component="RidgeForecaster (sklearn.linear_model.Ridge)",
        module="baselines.py",
        status="library",
        reference=(
            "A. E. Hoerl and R. W. Kennard, "
            '"Ridge regression: Biased estimation for nonorthogonal problems," '
            "Technometrics, vol. 12, no. 1, pp. 55-67, 1970; F. Pedregosa et al., "
            '"Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., '
            "vol. 12, pp. 2825-2830, 2011."
        ),
        repo="https://github.com/scikit-learn/scikit-learn",
        licence="BSD-3-Clause",
        accessed="2026-09-23",
        adapted=(
            "Ridge(fit_intercept=True, solver='cholesky') on float64 flattened "
            "K x 96 windows, one fit per alpha; alpha is chosen on validation MSE "
            "and the fitted coefficients are copied into a torch module."
        ),
    ),
    Upstream(
        component="Naive-RW benchmark (OriginTensors.naive_rw_z)",
        module="splits.py",
        status="own",
        reference=(
            "R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and "
            "Practice, 3rd ed. Melbourne, Australia: OTexts, 2021."
        ),
        adapted=(
            "A random walk in price predicts a zero log-return. In scaler space "
            "that is -mu_g/sigma_g, not 0, which would be the training drift."
        ),
    ),
    Upstream(
        component="walk-forward with purging (Origin, build_origin_tensors)",
        module="splits.py",
        status="own",
        reference=(
            "M. Lopez de Prado, Advances in Financial Machine Learning. "
            "Hoboken, NJ: Wiley, 2018, ch. 7; L. J. Tashman, "
            '"Out-of-sample tests of forecasting accuracy: An analysis and '
            'review," Int. J. Forecast., vol. 16, no. 4, pp. 437-450, 2000; '
            "C. Bergmeir and J. M. Benitez, "
            '"On the use of cross-validation for time series predictor '
            'evaluation," Information Sciences, vol. 191, pp. 192-213, 2012.'
        ),
        adapted=(
            "Purging at both boundaries (train/validation and train/test); no "
            "embargo, because every feature is per-bar; origins five months apart."
        ),
    ),
    Upstream(
        component="Scaler",
        module="splits.py",
        status="own",
        reference="No upstream publication: a per-channel z-score.",
        adapted="Fitted on the 21-month training sub-block only, at every origin.",
    ),
    Upstream(
        component="Adam optimiser and StepLR schedule (train_one)",
        module="train.py",
        status="library",
        reference=(
            "D. P. Kingma and J. Ba, "
            '"Adam: A method for stochastic optimization," in Proc. 3rd Int. '
            "Conf. Learn. Represent. (ICLR), 2015. arXiv:1412.6980."
        ),
        repo="https://docs.pytorch.org/docs/stable/optim.html",
        licence="BSD-3-Clause (PyTorch)",
        accessed="2026-09-03",
        adapted=(
            "Adam at lr 1e-4; StepLR halves every four epochs so the 30-epoch cap "
            "can bind. The loop is written here: GPU-resident splits, index "
            "slicing, no Dataset or DataLoader."
        ),
    ),
    Upstream(
        component="dm_test, clark_west_test (HLN correction, rectangular LRV)",
        module="metrics.py",
        status="own",
        reference=(
            "F. X. Diebold and R. S. Mariano, "
            '"Comparing predictive accuracy," J. Bus. Econ. Statist., vol. 13, '
            "no. 3, pp. 253-263, 1995; D. Harvey, S. Leybourne, and P. Newbold, "
            '"Testing the equality of prediction mean squared errors," Int. J. '
            "Forecast., vol. 13, no. 2, pp. 281-291, 1997; T. E. Clark and "
            'K. D. West, "Approximately normal tests for equal predictive '
            'accuracy in nested models," J. Econometrics, vol. 138, no. 1, '
            "pp. 291-311, 2007."
        ),
        adapted=(
            "Written on numpy. The long-run variance is the truncated rectangular "
            "estimator at lag h-1, not Bartlett, which would shrink the high-lag "
            "autocovariances. Clark-West is applied only against Naive-RW."
        ),
    ),
    Upstream(
        component="wild cluster restricted bootstrap (WCR) for beta1",
        module="metrics.py",
        status="own",
        reference=(
            "A. C. Cameron, J. B. Gelbach, and D. L. Miller, "
            '"Bootstrap-based improvements for inference with clustered errors," '
            "Rev. Econ. Statist., vol. 90, no. 3, pp. 414-427, 2008; "
            "J. G. MacKinnon, M. O. Nielsen, and M. D. Webb, "
            '"Cluster-robust inference: A guide to empirical practice," '
            "J. Econometrics, vol. 232, no. 2, pp. 272-299, 2023."
        ),
        adapted=(
            "Restricted (the null is imposed), bootstrapping the cluster-robust t, "
            "with Rademacher and Webb weights both reported and p = (1 + count)/(1 + B)."
        ),
    ),
    Upstream(
        component="romano_wolf",
        module="comparisons.py",
        status="own",
        reference=(
            "J. P. Romano and M. Wolf, "
            '"Stepwise multiple testing as formalized data snooping," '
            "Econometrica, vol. 73, no. 4, pp. 1237-1282, 2005."
        ),
        adapted="Stepdown over all pairs and within each declared family.",
    ),
    Upstream(
        component="model_confidence_set, mcs_table",
        module="comparisons.py",
        status="own",
        reference=(
            "P. R. Hansen, A. Lunde, and J. M. Nason, "
            '"The model confidence set," Econometrica, vol. 79, no. 2, '
            "pp. 453-497, 2011."
        ),
        adapted="Reported at 90% and 75% as a membership column of Table 6.",
    ),
    Upstream(
        component="variance_ratio and adf inside efficiency_table",
        module="efficiency.py",
        status="library",
        reference=(
            "A. W. Lo and A. C. MacKinlay, "
            '"Stock market prices do not follow random walks: Evidence from a '
            'simple specification test," Rev. Financial Stud., vol. 1, no. 1, '
            "pp. 41-66, 1988."
        ),
        repo="https://github.com/bashtage/arch",
        licence="NCSA (arch), BSD-3-Clause (statsmodels)",
        accessed="2026-09-03",
        adapted=(
            "arch.unitroot.VarianceRatio and statsmodels.tsa.stattools.adfuller "
            "are called directly, imported inside the function that needs them."
        ),
    ),
    Upstream(
        component="hurst_rs",
        module="efficiency.py",
        status="own",
        reference=(
            "H. E. Hurst, "
            '"Long-term storage capacity of reservoirs," Trans. Amer. Soc. Civil '
            "Eng., vol. 116, no. 1, pp. 770-799, 1951."
        ),
        adapted="Rescaled-range estimator written here on numpy.",
    ),
    Upstream(
        component="participation_ratio, stable_rank, lookback_correlation_pr",
        module="keff.py",
        status="own",
        reference=(
            "L. Laloux, P. Cizeau, J.-P. Bouchaud, and M. Potters, "
            '"Noise dressing of financial correlation matrices," Phys. Rev. '
            "Lett., vol. 83, no. 7, pp. 1467-1470, 1999; V. Plerou, "
            "P. Gopikrishnan, B. Rosenow, L. A. N. Amaral, T. Guhr, and "
            'H. E. Stanley, "Random matrix approach to cross correlations in '
            'financial data," Phys. Rev. E, vol. 65, no. 6, 066126, 2002.'
        ),
        adapted=(
            "PR on correlation matrices, never covariance, because the variates "
            "do not share units; measured per origin on the training sub-block."
        ),
    ),
    Upstream(
        component="parkinson, garman_klass, rogers_satchell (family F2)",
        module="features.py",
        status="own",
        reference=(
            "M. Parkinson, "
            '"The extreme value method for estimating the variance of the rate '
            'of return," J. Business, vol. 53, no. 1, pp. 61-65, 1980; '
            'M. B. Garman and M. J. Klass, "On the estimation of security price '
            'volatilities from historical data," J. Business, vol. 53, no. 1, '
            "pp. 67-78, 1980; L. C. G. Rogers and S. E. Satchell, "
            '"Estimating variance from high, low and closing prices," Ann. Appl. '
            "Probab., vol. 1, no. 4, pp. 504-512, 1991."
        ),
        adapted=(
            "Per-bar, never smoothed. Rogers-Satchell vanishes on shadowless "
            "bars, so it is taken as log(RS + 1e-9)."
        ),
    ),
)
