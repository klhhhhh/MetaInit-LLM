import pytorch_lightning as pl

class ProjectorScheduleAndFreeze(pl.Callback):
    def __init__(self, begin_step=0, end_step=800, freeze_at=800, set_alpha_to=0.0, cache=True, drop_small=True):
        super().__init__()
        self.begin_step = begin_step
        self.end_step = end_step
        self.freeze_at = freeze_at
        self.set_alpha_to = set_alpha_to
        self.cache = cache
        self.drop_small = drop_small
        self._frozen = False

    @staticmethod
    def _iter_proj_modules(model):
        for m in model.modules():
            if isinstance(m, (ColumnParallelLinearWithProjector, RowParallelLinearWithProjector)):
                yield m

    def on_train_start(self, trainer, pl_module):
        # 可选：统一设置 α 调度（起始 α 用层当前值）
        for m in self._iter_proj_modules(pl_module):
            m.set_alpha_schedule(begin_step=self.begin_step, end_step=self.end_step, end=0.0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gs = trainer.global_step
        # 更新 α
        for m in self._iter_proj_modules(pl_module):
            m.update_alpha_by_step(gs)

        # 达到 freeze_at：冻结并（可选）删除 W_small
        if (not self._frozen) and (gs >= self.freeze_at):
            for m in self._iter_proj_modules(pl_module):
                m.freeze_projector(cache=self.cache, set_alpha_to=self.set_alpha_to, drop_small=self.drop_small)
            self._frozen = True
