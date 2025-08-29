import math
import pytorch_lightning as pl
import torch

class AlphaScheduleAndFreeze(pl.Callback):
    """
    - Perform linear/cosine scheduling of alpha_mult for all *WithProjector layers 
      within the [start_step, end_step] range. Supports 'decay' (from start_value -> end_value) 
      or 'warmup' (reverse).
    - When global_step >= freeze_at, call freeze_projector() and stop further scheduling.
    - After freezing, alpha_mult is no longer modified to avoid implicit interpolation.
    """
    def __init__(
        self,
        schedule: str = "cosine",      # ["cosine","linear","none"]
        mode: str = "decay",           # ["decay","warmup"]
        start_value: float = 1.0,
        end_value: float = 0.05,
        start_step: int = 0,
        end_step: int = 10000,
        freeze_at: int | None = None,
        cache_on_freeze: bool = True,
        drop_small_on_freeze: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        assert schedule in ("cosine", "linear", "none")
        assert mode in ("decay", "warmup")
        self.schedule = schedule
        self.mode = mode
        self.start_value = float(start_value)
        self.end_value   = float(end_value)
        self.start_step  = int(start_step)
        self.end_step    = int(end_step)
        self.freeze_at   = None if freeze_at is None else int(freeze_at)
        self.cache_on_freeze = bool(cache_on_freeze)
        self.drop_small_on_freeze = bool(drop_small_on_freeze)
        self.verbose = verbose
        self._frozen_done = False

    @staticmethod
    def _iter_proj_modules(model):
        from projection.layer_projection import (
            ColumnParallelLinearWithProjector,
            RowParallelLinearWithProjector,
        )
        for m in model.modules():
            if isinstance(m, (ColumnParallelLinearWithProjector, RowParallelLinearWithProjector)):
                yield m

    def _compute_mult(self, step: int) -> float:
        if self.schedule == "none":
            return self.start_value if self.mode == "decay" else self.end_value

        if step <= self.start_step:
            v0, v1, t = self.start_value, self.end_value, 0.0
        elif step >= self.end_step:
            v0, v1, t = self.start_value, self.end_value, 1.0
        else:
            t = (step - self.start_step) / max(1, (self.end_step - self.start_step))
            v0, v1 = self.start_value, self.end_value

        if self.mode == "warmup":
            # Reverse (end -> start)
            v0, v1 = self.end_value, self.start_value

        if self.schedule == "linear":
            val = v0 + (v1 - v0) * t
        else:  # cosine
            # Half-period cosine: smooth transition from v0 to v1
            cos_t = 0.5 * (1.0 - math.cos(math.pi * t))
            val = v0 + (v1 - v0) * cos_t

        # Clamp to [0,1], can be relaxed for >1 cases
        return max(0.0, min(1.0, float(val)))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step

        if self._frozen_done:
            return

        # If freezing is required and the freeze step is reached
        if (self.freeze_at is not None) and (not self._frozen_done) and (step >= self.freeze_at):
            for m in self._iter_proj_modules(pl_module):
                m.freeze_projector(cache=self.cache_on_freeze, drop_small=self.drop_small_on_freeze)
            self._frozen_done = True
            if self.verbose and trainer.is_global_zero:
                print(f"[AlphaScheduleAndFreeze] projector FROZEN at step={step}  (cache={self.cache_on_freeze}, drop_small={self.drop_small_on_freeze})")
            return  # Stop scheduling after freezing

        # Continue updating alpha_mult if not frozen
        mult = self._compute_mult(step)
        for m in self._iter_proj_modules(pl_module):
            # Skip updating if the layer is already individually frozen
            if int(getattr(m, "_proj_frozen", torch.tensor(0)).item()) == 1:
                continue
            m.set_alpha_multiplier(mult)

        if self.verbose and (step % 200 == 0) and trainer.is_global_zero:
            print(f"[AlphaScheduleAndFreeze] step={step}  alpha_mult={mult:.4f}")
