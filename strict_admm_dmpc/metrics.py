"""Ground-truth closed-loop metrics shared by every controller."""

from __future__ import annotations

import numpy as np

from strict_admm_dmpc.model import FloatArray


def formation_rmse(
    states: FloatArray, common_reference: FloatArray, offsets: FloatArray
) -> FloatArray:
    """Return time-wise formation RMSE against one shared true reference."""
    states = np.asarray(states, dtype=float)
    reference = np.asarray(common_reference, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if states.ndim != 3:
        raise ValueError("states must have shape (time, agent, state)")
    if offsets.ndim != 2 or offsets.shape[0] != states.shape[1]:
        raise ValueError("offsets must have one output row per agent")
    output_dimension = offsets.shape[1]
    if output_dimension < 1:
        raise ValueError("formation output dimension must be positive")
    if states.shape[2] < output_dimension:
        raise ValueError("states must have shape (time, agent, state)")
    if (
        reference.ndim != 2
        or reference.shape[0] != states.shape[0]
        or reference.shape[1] < output_dimension
    ):
        raise ValueError("reference must have one row per time")
    desired = reference[:, None, :output_dimension] + offsets[None, :, :]
    squared_distance = np.sum(
        (states[:, :, :output_dimension] - desired) ** 2, axis=2
    )
    return np.sqrt(np.mean(squared_distance, axis=1))


def sustained_settling_time(
    time: FloatArray,
    error: FloatArray,
    start_time: float,
    tolerance: float,
    residence_time: float,
) -> float:
    """First absolute time that begins a sustained interval inside tolerance."""
    time = np.asarray(time, dtype=float)
    error = np.asarray(error, dtype=float)
    if time.ndim != 1 or error.shape != time.shape or time.size < 2:
        raise ValueError("time and error must be matching one-dimensional arrays")
    if tolerance < 0.0 or residence_time < 0.0:
        raise ValueError("tolerance and residence_time must be nonnegative")
    for index, candidate in enumerate(time):
        if candidate < start_time or error[index] > tolerance:
            continue
        end = candidate + residence_time
        mask = (time >= candidate) & (time <= end + 1e-12)
        if time[mask].size and time[mask][-1] >= end - 1e-12:
            if np.all(error[mask] <= tolerance):
                return float(candidate)
    return float("nan")
