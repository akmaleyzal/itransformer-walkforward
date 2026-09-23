"""The two transformer models: the authors' code behind the study's training interface.

``itr`` and ``vtr`` are the ``Model`` classes of the official iTransformer
repository, copied unchanged (see ``upstream.py``). This module only adapts them.
It builds the ``configs`` namespace the upstream constructors read, passes
``x_mark=None`` because no calendar feature enters the study, gives the vanilla
decoder its start token, and reads the target channel ``r`` from the all-channel
output. The loss is MSE on that channel at every rung.

Upstream:
    Code: https://github.com/thuml/iTransformer (MIT), commit
    c2426e68ca13f74aaec08045c5c724d8ad328124. Y. Liu et al., "iTransformer:
    Inverted transformers are effective for time series forecasting," ICLR 2024;
    A. Vaswani et al., "Attention is all you need," NeurIPS 2017. Settings follow
    the official defaults (2 encoder layers, 1 decoder layer, gelu, label_len 48,
    8 heads, dropout 0.1), with d_model 128 and d_ff 256 for the sample size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from itransformer_btc.config import PRED_LEN, SEQ_LEN
from itransformer_btc.features import TARGET_INDEX
from itransformer_btc.upstream import upstream_model


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass(frozen=True, slots=True)
class ITransformerConfig:
    """iTransformer settings, named as the upstream ``configs`` fields.

    Fixed before any run and identical at every rung. The parameter count does
    not depend on K: K changes the number of tokens, not a weight shape.
    """

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
    factor: int = 1
    embed: str = "timeF"
    freq: str = "h"
    class_strategy: str = "projection"

    def upstream_configs(self) -> SimpleNamespace:
        """The ``configs`` object ``model/iTransformer.py`` reads."""
        return SimpleNamespace(**asdict(self), output_attention=False)

    def build(self) -> "ITransformerForecaster":
        return ITransformerForecaster(self)

    def schedule(self) -> "TrainSchedule":
        from itransformer_btc.train import TrainSchedule

        return TrainSchedule()

    def fit(self, tensors, spec, *, device=None):
        """Train one cell; nothing is selected, so the config comes back unchanged."""
        from itransformer_btc.train import train_one

        model, outcome = train_one(tensors, spec, self, device=device)
        return model, self, outcome


class ITransformerForecaster(nn.Module):
    """``(B, L, K) -> (B, H)``: the upstream iTransformer, read at the target channel."""

    def __init__(self, cfg: ITransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = upstream_model("itr")(cfg.upstream_configs())

    def forward(self, x: Tensor) -> Tensor:
        """Every channel's forecast, ``(B, H, K)``."""
        return self.model(x, None, None, None)

    def forecast_target(self, x: Tensor) -> Tensor:
        return self.forward(x)[:, :, TARGET_INDEX]

    def n_parameters(self) -> int:
        return _count(self)


@dataclass(frozen=True, slots=True)
class VanillaConfig:
    """Vanilla Transformer settings, named as the upstream ``configs`` fields.

    ``k`` is a field because the embeddings and the output head are K channels wide.
    """

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    label_len: int = 48
    k: int = 8
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 2
    d_layers: int = 1
    n_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    factor: int = 1
    embed: str = "timeF"
    freq: str = "h"

    def upstream_configs(self) -> SimpleNamespace:
        """The ``configs`` object ``model/Transformer.py`` reads."""
        return SimpleNamespace(
            **asdict(self), enc_in=self.k, dec_in=self.k, c_out=self.k,
            channel_independence=False, output_attention=False,
        )

    def build(self) -> "VanillaForecaster":
        return VanillaForecaster(self)

    def schedule(self) -> "TrainSchedule":
        from itransformer_btc.train import TrainSchedule

        return TrainSchedule()

    def fit(self, tensors, spec, *, device=None):
        """Train one cell; nothing is selected, so the config comes back unchanged."""
        from itransformer_btc.train import train_one

        model, outcome = train_one(tensors, spec, self, device=device)
        return model, self, outcome


class VanillaForecaster(nn.Module):
    """``(B, L, K) -> (B, H)``: the upstream vanilla Transformer, read at the target channel.

    The decoder input is the last ``label_len`` lookback hours followed by
    ``pred_len`` zero placeholders, so every hour it reads is known at the
    forecast origin and all 24 hours are decoded in one pass.
    """

    def __init__(self, cfg: VanillaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = upstream_model("vtr")(cfg.upstream_configs())

    def decoder_input(self, x: Tensor) -> Tensor:
        """``(B, label_len + pred_len, K)``: the known start token, then zeros."""
        placeholders = x.new_zeros(x.shape[0], self.cfg.pred_len, x.shape[2])
        return torch.cat([x[:, -self.cfg.label_len:, :], placeholders], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Every channel's forecast, ``(B, H, K)``."""
        return self.model(x, None, self.decoder_input(x), None)

    def forecast_target(self, x: Tensor) -> Tensor:
        return self.forward(x)[:, :, TARGET_INDEX]

    def n_parameters(self) -> int:
        return _count(self)
