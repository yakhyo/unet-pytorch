import torch

from typing import Dict, Any


class Scheduler:

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            if 'lr_scale' in param_group:
                param_group[self.param_group_field] = value * param_group['lr_scale']
            else:
                param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lrs = [v + v * noise for v in lrs]
        return lrs

    def _is_apply_noise(self, t) -> bool:
        """Return True if scheduler in noise range."""
        apply_noise = False
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
        return apply_noise

    def _calculate_noise(self, t) -> float:
        g = torch.Generator()
        g.manual_seed(self.noise_seed + t)
        if self.noise_type == 'normal':
            while True:
                # resample if noise out of percent limit, brute force but shouldn't spin much
                noise = torch.randn(1, generator=g).item()
                if abs(noise) < self.noise_pct:
                    return noise
        else:
            noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
        return noise


class PlateauLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,
                 decay_rate=0.1,
                 patience_t=10,
                 verbose=True,
                 threshold=1e-4,
                 cooldown_t=0,
                 warmup_t=0,
                 warmup_lr_init=0,
                 lr_min=0,
                 mode='max',
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize=True,
                 ):
        super().__init__(
            optimizer,
            'lr',
            noise_range_t=noise_range_t,
            noise_type=noise_type,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=patience_t,
            factor=decay_rate,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown_t,
            mode=mode,
            min_lr=lr_min
        )

        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.restore_lr = None

    def state_dict(self):
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    # override the base class step fn completely
    def step(self, epoch, metric=None):
        if epoch <= self.warmup_t:
            lrs = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
            super().update_groups(lrs)
        else:
            if self.restore_lr is not None:
                # restore actual LR from before our last noise perturbation before stepping base
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.restore_lr[i]
                self.restore_lr = None

            self.lr_scheduler.step(epoch)  # step the base scheduler

            if self._is_apply_noise(epoch):
                self._apply_noise(epoch)

    def _apply_noise(self, epoch):
        noise = self._calculate_noise(epoch)

        # apply the noise on top of previous LR, cache the old value so we can restore for normal
        # stepping of base scheduler
        restore_lr = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            restore_lr.append(old_lr)
            new_lr = old_lr + old_lr * noise
            param_group['lr'] = new_lr
        self.restore_lr = restore_lr
