from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - exercised only when gymnasium is installed.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback is covered in local tests.
    class Space:
        def sample(self) -> Any:
            raise NotImplementedError

    class Discrete(Space):
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(0, self.n))

    class Box(Space):
        def __init__(self, low: float, high: float, shape: Iterable[int], dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class Dict(Space):
        def __init__(self, spaces_dict: dict[str, Space]):
            self.spaces = spaces_dict

        def sample(self) -> dict[str, Any]:
            return {key: space.sample() for key, space in self.spaces.items()}

    class Env:
        observation_space: Space
        action_space: Space

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            raise NotImplementedError

        def step(self, action: Any):
            raise NotImplementedError

    class _FallbackGym:
        Env = Env

    class _FallbackSpaces:
        Box = Box
        Dict = Dict
        Discrete = Discrete

    gym = _FallbackGym()
    spaces = _FallbackSpaces()
