from abc import ABC
from abc import abstractmethod

import jax
import jax.numpy as jnp
import ring
from ring.utils import pickle_load
from ring.utils import pickle_save
import tree_utils


def _to_3d(tree):
    if tree is None:
        return None
    return jax.tree_map(lambda arr: arr[None], tree)


def _to_2d(tree, i: int = 0):
    if tree is None:
        return None
    return jax.tree_map(lambda arr: arr[i], tree)


class AbstractFilter(ABC):
    def _apply_unbatched(self, X, params, state, y, lam):
        return _to_2d(
            self._apply_batched(
                X=_to_3d(X), params=params, state=_to_3d(state), y=_to_3d(y), lam=lam
            )
        )

    @abstractmethod
    def _apply_batched(self, X, params, state, y, lam):
        pass

    @abstractmethod
    def init(self, bs, X, lam, seed: int):
        pass

    def apply(self, X, params=None, state=None, y=None, lam=None):
        "X.shape = (B, T, N, F) or (T, N, F)"
        assert X.ndim in [3, 4]
        if X.ndim == 4:
            return self._apply_batched(X, params, state, y, lam)
        else:
            return self._apply_unbatched(X, params, state, y, lam)

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            raise NotImplementedError

        if self._name is None:
            raise RuntimeError("No `name` was given.")
        return self._name

    def nojit(self) -> "AbstractFilter":
        return self

    def _pre_save(self, *args, **kwargs) -> None:
        pass

    def save(self, path: str, *args, **kwargs):
        self._pre_save(*args, **kwargs)
        pickle_save(self.nojit(), path, overwrite=True)

    @staticmethod
    def _post_load(filter: "AbstractFilter", *args, **kwargs) -> "AbstractFilter":
        pass

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        filter = pickle_load(path)
        return cls._post_load(filter, *args, **kwargs)

    def search_attr(self, attr: str):
        return getattr(self, attr)


class AbstractFilterUnbatched(AbstractFilter):
    @abstractmethod
    def _apply_unbatched(self, X, params, state, y, lam):
        pass

    def _apply_batched(self, X, params, state, y, lam):
        N = X.shape[0]
        ys = []
        for i in range(N):
            ys.append(
                self._apply_unbatched(
                    _to_2d(X, i), params, _to_2d(state, i), _to_2d(y, i), lam
                )
            )
        return tree_utils.tree_batch(ys)


class AbstractFilterWrapper(AbstractFilter):
    def __init__(self, filter: AbstractFilter, name=None) -> None:
        self._filter = filter
        self._name = name

    def _apply_batched(self, X, params, state, y, lam):
        raise NotImplementedError

    @property
    def unwrapped(self) -> AbstractFilter:
        return self._filter

    def apply(self, X, params=None, state=None, y=None, lam=None):
        return self.unwrapped.apply(X=X, params=params, state=state, y=y, lam=lam)

    def init(self, bs=None, X=None, lam=None, seed: int = 1):
        return self.unwrapped.init(bs=bs, X=X, lam=lam, seed=seed)

    def nojit(self) -> "AbstractFilterWrapper":
        self._filter = self.unwrapped.nojit()
        return self

    def search_attr(self, attr: str):
        if hasattr(self, attr):
            return super().search_attr(attr)
        return self.unwrapped.search_attr(attr)

    def _pre_save(self, *args, **kwargs):
        self.unwrapped._pre_save(*args, **kwargs)

    @staticmethod
    def _post_load(
        wrapper: "AbstractFilterWrapper", *args, **kwargs
    ) -> "AbstractFilterWrapper":
        wrapper._filter = wrapper._filter._post_load(wrapper._filter, *args, **kwargs)
        return wrapper

    @property
    def name(self):
        return self.unwrapped.name + " ->\n" + super().name


class LPF_FilterWrapper(AbstractFilterWrapper):
    def __init__(
        self,
        filter: AbstractFilter,
        cutoff_freq: float,
        samp_freq: float | None,
        filtfilt: bool = True,
        name="LPF_FilterWrapper",
    ) -> None:
        super().__init__(filter, name)
        self.samp_freq = samp_freq
        self._kwargs = dict(cutoff_freq=cutoff_freq, filtfilt=filtfilt)

    def apply(self, X, params=None, state=None, y=None, lam=None):
        if X.ndim == 4:
            if self.samp_freq is not None:
                samp_freq = jnp.repeat(jnp.array(self.samp_freq), X.shape[0])
            else:
                assert X.shape[-1] == 10
                dt = X[:, 0, 0, -1]
                samp_freq = 1 / dt
        else:
            if self.samp_freq is not None:
                samp_freq = jnp.array(self.samp_freq)
            else:
                assert X.shape[-1] == 10
                dt = X[0, 0, -1]
                samp_freq = 1 / dt

        if self.samp_freq is None:
            print(f"Detected the following sampling rates from `X`: {samp_freq}")

        yhat, state = super().apply(X, params, state, y, lam)

        if yhat.ndim == 4:
            yhat = jax.vmap(
                jax.vmap(
                    lambda q, samp_freq: ring.maths.quat_lowpassfilter(
                        q, samp_freq=samp_freq, **self._kwargs
                    ),
                    in_axes=(2, None),
                    out_axes=2,
                )
            )(yhat, samp_freq)
        else:
            yhat = jax.vmap(
                lambda q, samp_freq: ring.maths.quat_lowpassfilter(
                    q, samp_freq=samp_freq, **self._kwargs
                ),
                in_axes=(1, None),
                out_axes=1,
            )(yhat, samp_freq)
        return yhat, state


class GroundTruthHeading_FilterWrapper(AbstractFilterWrapper):

    def __init__(
        self, filter: AbstractFilter, name="GroundTruthHeading_FilterWrapper"
    ) -> None:
        super().__init__(filter, name)

    def apply(self, X, params=None, state=None, y=None, lam=None):
        yhat, state = super().apply(X, params, state, y, lam)
        if lam is None:
            lam = self.search_attr("lam")
        yhat = self.transfer_ground_truth_heading(lam, y, yhat)
        return yhat, state

    @staticmethod
    def transfer_ground_truth_heading(lam, y, yhat) -> None:
        if y is None or lam is None:
            return yhat

        yhat = jnp.array(yhat)
        for i, p in enumerate(lam):
            if p == -1:
                yhat = yhat.at[..., i, :].set(
                    ring.maths.quat_transfer_heading(y[..., i, :], yhat[..., i, :])
                )
        return yhat


_default_factors = dict(gyr=1 / 2.2, acc=1 / 9.81, joint_axes=1 / 0.57, dt=10.0)


class ScaleX_FilterWrapper(AbstractFilterWrapper):

    def __init__(
        self,
        filter: AbstractFilter,
        factors: dict[str, float] = _default_factors,
        name="ScaleX_FilterWrapper",
    ) -> None:
        super().__init__(filter, name)
        self._factors = factors

    def apply(self, X, params=None, state=None, y=None, lam=None):
        F = X.shape[-1]
        num_batch_dims = X.ndim - 1

        if F == 6:
            X = dict(acc=X[..., :3], gyr=X[..., 3:])
        elif F == 9:
            X = dict(acc=X[..., :3], gyr=X[..., 3:6], joint_axes=X[..., 6:])
        elif F == 10:
            X = dict(
                acc=X[..., :3], gyr=X[..., 3:6], joint_axes=X[..., 6:9], dt=X[..., 9:10]
            )
        else:
            raise Exception(f"X.shape={X.shape}")
        X = {key: val * self._factors[key] for key, val in X.items()}
        X = tree_utils.batch_concat_acme(X, num_batch_dims=num_batch_dims)
        return super().apply(X, params, state, y, lam)
