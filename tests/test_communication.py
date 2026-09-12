from __future__ import annotations

import numpy as np

from strict_admm_dmpc.communication import (
    DenialSchedule,
    MessageCache,
    PredictionMode,
)


def test_denial_schedule_uses_half_open_interval() -> None:
    schedule = DenialSchedule(start=2.0, end=5.0)

    assert not schedule.is_denied(1.999)
    assert schedule.is_denied(2.0)
    assert schedule.is_denied(4.999)
    assert not schedule.is_denied(5.0)


def test_cache_does_not_refresh_during_denial() -> None:
    schedule = DenialSchedule(start=1.0, end=3.0)
    cache = MessageCache(n_agents=2, schedule=schedule)
    states_0 = np.array([[0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0]])
    cache.observe(0.5, states_0)
    cache.observe(1.5, states_0 + 100.0)

    prediction = cache.predict_aligned(
        agent_id=1,
        time=1.5,
        horizon=2,
        dt=0.5,
        offset=np.array([1.0, 0.0]),
        mode=PredictionMode.CONSTANT_VELOCITY,
    )

    np.testing.assert_allclose(prediction.trajectory[:, 0], [2.0, 2.5, 3.0])
    assert prediction.age == 1.0


def test_constant_velocity_and_hold_diverge_for_moving_agent() -> None:
    schedule = DenialSchedule(start=1.0, end=3.0)
    cache = MessageCache(n_agents=1, schedule=schedule)
    cache.observe(0.0, np.array([[0.0, 0.0, 2.0, -1.0]]))

    hold = cache.predict_aligned(
        0, 2.0, 2, 0.5, np.zeros(2), PredictionMode.ZERO_ORDER_HOLD
    )
    velocity = cache.predict_aligned(
        0, 2.0, 2, 0.5, np.zeros(2), PredictionMode.CONSTANT_VELOCITY
    )

    np.testing.assert_allclose(hold.trajectory, np.zeros((3, 2)))
    np.testing.assert_allclose(
        velocity.trajectory,
        np.array([[4.0, -2.0], [5.0, -2.5], [6.0, -3.0]]),
    )


def test_exponential_confidence_decreases_with_message_age() -> None:
    schedule = DenialSchedule(start=1.0, end=5.0)
    cache = MessageCache(n_agents=1, schedule=schedule, confidence_time_constant=2.0)
    cache.observe(0.0, np.zeros((1, 4)))

    early = cache.predict_aligned(
        0, 1.0, 1, 0.1, np.zeros(2), PredictionMode.ZERO_ORDER_HOLD
    )
    late = cache.predict_aligned(
        0, 3.0, 1, 0.1, np.zeros(2), PredictionMode.ZERO_ORDER_HOLD
    )

    assert 0.0 < late.confidence < early.confidence < 1.0


def test_cache_supports_scalar_double_integrator_layout() -> None:
    schedule = DenialSchedule(start=1.0, end=3.0)
    cache = MessageCache(
        n_agents=1,
        schedule=schedule,
        state_dimension=2,
        output_dimension=1,
    )
    cache.observe(0.0, np.array([[2.0, 0.5]]))

    prediction = cache.predict_aligned(
        0, 2.0, 2, 0.5, np.array([1.0]), PredictionMode.CONSTANT_VELOCITY
    )

    np.testing.assert_allclose(prediction.trajectory[:, 0], [2.0, 2.25, 2.5])
