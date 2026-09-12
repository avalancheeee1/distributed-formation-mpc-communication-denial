"""Strict edge-splitting ADMM-DMPC reference implementation."""

from strict_admm_dmpc.model import (
    FormationGraph,
    LinearDynamics,
    MPCConfig,
    MPCWeights,
    Scenario,
    build_double_integrator,
)

__all__ = [
    "FormationGraph",
    "LinearDynamics",
    "MPCConfig",
    "MPCWeights",
    "Scenario",
    "build_double_integrator",
]

