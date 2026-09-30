"""Step-wise linear warmup and cosine decay, preserving parameter-group LR ratios."""
import math


class WarmupCosine:
    def __init__(self, optimizer, total_steps, warmup_steps, min_factor=0.01):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError('warmup_steps must be smaller than total_steps')
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_factor = min_factor
        self.step_index = 0
        self._apply()

    def factor(self):
        if self.step_index < self.warmup_steps:
            return (self.step_index + 1) / max(1, self.warmup_steps)
        progress = min(1., (self.step_index - self.warmup_steps) /
                       max(1, self.total_steps - self.warmup_steps - 1))
        return self.min_factor + (1 - self.min_factor) * (1 + math.cos(math.pi * progress)) / 2

    def _apply(self):
        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = lr * self.factor()

    def step(self):
        self.step_index += 1
        self._apply()

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k != 'optimizer'}

    def load_state_dict(self, state):
        self.__dict__.update(state)
        self._apply()
