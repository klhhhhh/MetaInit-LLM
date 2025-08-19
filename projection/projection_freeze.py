import pytorch_lightning as pl

class FreezeProjectorAtStep(pl.Callback):
    """
    When global_step >= freeze_at, for all *WithProjector layers:
      - Freeze A/B and alpha (or alpha_logit)
      - Optionally cache W_proj_scaled once
      - Optionally release small model weights
    """
    def __init__(self, *, freeze_at: int = 1000, cache: bool = True, drop_small: bool = True):
        super().__init__()
        self.freeze_at = int(freeze_at)
        self.cache = cache
        self.drop_small = drop_small
        self._done = False

    @staticmethod
    def _iter_proj_modules(model):
        from projection.layer_projection import (
            ColumnParallelLinearWithProjector,
            RowParallelLinearWithProjector,
        )
        for m in model.modules():
            if isinstance(m, (ColumnParallelLinearWithProjector, RowParallelLinearWithProjector)):
                yield m

    def on_train_start(self, trainer, pl_module):
        print(">>>FreezeProjectorAtStep Callback triggered on_train_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._done:
            return
        gs = trainer.global_step
        if gs >= self.freeze_at:
            for m in self._iter_proj_modules(pl_module):
                m.freeze_projector(cache=self.cache, drop_small=self.drop_small)
            self._done = True
            if trainer.is_global_zero:
                print(f"[FreezeProjectorAtStep] projector frozen at step={gs} (cache={self.cache}, drop_small={self.drop_small})")
