"""Explicit message-age, prediction, and confidence models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from strict_admm_dmpc.model import FloatArray


class PredictionMode(str, Enum):
    """Fallback used when current neighbor messages are unavailable."""

    ZERO_ORDER_HOLD = "zero_order_hold"
    CONSTANT_VELOCITY = "constant_velocity"


@dataclass(frozen=True)
class DenialSchedule:
    """A deterministic half-open full-denial interval."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError("denial times must be finite")
        if self.start < 0.0 or self.end <= self.start:
            raise ValueError("denial interval must satisfy 0 <= start < end")

    def is_denied(self, time: float) -> bool:
        return self.start <= time < self.end


@dataclass(frozen=True)
class PredictedMessage:
    trajectory: FloatArray
    age: float
    confidence: float


class MessageCache:
    """Last-received agent states; denied observations never refresh the cache."""

    def __init__(
        self,
        n_agents: int,
        schedule: DenialSchedule,
        confidence_time_constant: float = 12.0,
        state_dimension: int = 4,
        output_dimension: int = 2,
    ) -> None:
        if not isinstance(n_agents, int) or isinstance(n_agents, bool) or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        if (
            not isinstance(state_dimension, int)
            or isinstance(state_dimension, bool)
            or not isinstance(output_dimension, int)
            or isinstance(output_dimension, bool)
            or output_dimension < 1
            or state_dimension < 2 * output_dimension
        ):
            raise ValueError("state layout must contain output position and velocity")
        if not np.isfinite(confidence_time_constant) or confidence_time_constant <= 0.0:
            raise ValueError("confidence_time_constant must be finite and positive")
        self._schedule = schedule
        self._time_constant = confidence_time_constant
        self._output_dimension = output_dimension
        self._states = np.full((n_agents, state_dimension), np.nan)
        self._times = np.full(n_agents, np.nan)

    def observe(self, time: float, states: FloatArray) -> None:
        """Store an actual received message only while communication is active."""
        states = np.asarray(states, dtype=float)
        if not np.isfinite(time):
            raise ValueError("observation time must be finite")
        if states.shape != self._states.shape or not np.all(np.isfinite(states)):
            raise ValueError("observed states have incompatible dimensions")
        finite_times = self._times[np.isfinite(self._times)]
        if finite_times.size and time < float(np.max(finite_times)):
            raise ValueError("observation time must be monotone")
        if self._schedule.is_denied(time):
            return
        self._states[:] = states
        self._times[:] = time

    def predict_aligned(
        self,
        agent_id: int,
        time: float,
        horizon: int,
        dt: float,
        offset: FloatArray,
        mode: PredictionMode,
    ) -> PredictedMessage:
        """Predict a formation-aligned position trajectory from the last message."""
        if not 0 <= agent_id < self._states.shape[0]:
            raise ValueError("agent_id is out of range")
        if not np.isfinite(self._times[agent_id]):
            raise RuntimeError(f"agent {agent_id} has no cached message")
        if not np.isfinite(time) or time < self._times[agent_id]:
            raise ValueError("prediction time must be finite and not precede the cache")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (self._output_dimension,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset has incompatible dimensions")
        age = float(time - self._times[agent_id])
        state = self._states[agent_id]
        future = age + np.arange(horizon + 1, dtype=float) * dt
        if mode is PredictionMode.ZERO_ORDER_HOLD:
            position = np.repeat(
                state[None, : self._output_dimension], horizon + 1, axis=0
            )
        elif mode is PredictionMode.CONSTANT_VELOCITY:
            position = (
                state[None, : self._output_dimension]
                + future[:, None]
                * state[None, self._output_dimension : 2 * self._output_dimension]
            )
        else:
            raise ValueError(f"unsupported prediction mode: {mode}")
        confidence = float(np.exp(-age / self._time_constant))
        return PredictedMessage(
            trajectory=position - offset,
            age=age,
            confidence=confidence,
        )
