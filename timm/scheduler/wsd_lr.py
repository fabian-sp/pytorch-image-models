""" WSD Scheduler

WSD schedule (warmup-constant-cooldown).

Hacked together by FS
"""
import math
import torch
from typing import List


from .scheduler import Scheduler

class WSDScheduler(Scheduler):
    """
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            cooldown_t: float,
            final_lr: float=0.,
            total_t: int=100,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=True,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.cooldown_t = cooldown_t # start iter of cooldown
        self.total_t = total_t
        assert total_t > cooldown_t, "Total length must be larger than cooldown start"
        self.final_lr = final_lr
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t
            if t < self.cooldown_t:
                lrs = [v for v in self.base_values] # constant
            else:
                # t and cooldown_t are 0-indexed, total_t is 1-indexed
                # such that in final epoch we still did not reach zero (but would in next step)
                a_t = (t-self.cooldown_t)/(self.total_t - self.cooldown_t)
                lrs = [v * (1 - a_t*(1-self.final_lr)) for v in self.base_values] # cooldown
        return lrs
