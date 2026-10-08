"""Certify the analytic Lipschitz constant that replaces the E8 estimate 0.94.

The E8 protocol reports ``L = 0.94``, obtained by chord-differencing the
minimizer along the uniform weight path ``w = s * 1`` at ``s = 1``, and
acknowledges it as a lower bound rather than the sharp constant.  This script
closes the dependence analytically, and checks each of its steps numerically
against the same protocol (chain/ring/connected_random, 30 seeds, horizon 10)
whose raw output is committed at ``artifacts_v3_extra/raw/e8_lipschitz.json``.

Three things are established here, each checked numerically rather than
asserted.

**1. The strong-convexity modulus is the input cost, by construction.**
Condensing the dynamics into the control sequence, the uncoupled Hessian is

    H = blockdiag(R) + sum_k Phi_k^T Q Phi_k + Phi_N^T P Phi_N  >=  blockdiag(R),

since every term but ``blockdiag(R)`` is positive semidefinite.  So the modulus
is ``mu = lambda_min(R) = 0.15`` for *every* seed, horizon and topology -- it is
read off the weights, not measured.  Check 1 confirms the assembled finite
matrix respects it.

**2. The confidence decay is an explicit convex perturbation.**
The coupling term of the centralized cost is ``(gamma/2) sum_e w_e ||G_e u||^2``,
whose gradient is ``gamma M(w) u`` with ``M(w) = sum_e w_e M_e``.  On a fixed
active set ``S`` the two stationarity conditions give the *exact* identity

    u*(w) - u*_loc = -gamma (H_SS + gamma M_SS(w))^{-1} (M(w) u*_loc)_S,

so the constant is explicitly

    L = gamma * max_{||w||_1 <= 1} || [Phi; I] (H_SS + gamma M_SS)^{-1} (M(w) u*_loc)_S || .

Dropping the ``gamma M_SS`` term and bounding ``||M(w) u*_loc||`` by the
worst single edge gives the unconditional, inflation-free upper bound

    L <= gamma * max_e ||M_e u*_loc|| / lambda_min(R)          (Check 2)

while evaluating the identity along the uniform path gives the sharp value
(Check 3).

**3. The submitted protocol is biased low, not merely loose.**
E8 reads the ratio off the chord at ``s = 1``.  The map ``s -> x*(s*1)`` is
concave enough that this chord underpredicts the ``s -> 0`` tangent by up to a
factor of two, so the reported 0.94 is not just "a lower bound" -- it is a
*systematically* low estimate, and the paper's own bound is correspondingly
optimistic.  Check 4 quantifies the bias on the E8 protocol.

Run::

    python validate_lipschitz_analytic.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from strict_admm_dmpc.centralized_qp import solve_centralized_qp
from strict_admm_dmpc.experiments import build_scenario

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts_v3_rev1" / "tables" / "lipschitz_analytic.json"

#: The E8 protocol, reproduced exactly so the numbers are comparable.
TOPOLOGIES = ("chain", "ring", "connected_random")
SEEDS = range(30)
HORIZON = 10
E8_SCALES = (0.2, 0.4, 0.6, 0.8, 1.0)
SMALL_S = 1e-5


def _uncoupled(scenario):
    n_edges = len(scenario.graph.edges)
    return replace(
        scenario, graph=replace(scenario.graph, edge_weights=np.zeros(n_edges))
    )


def _decision(states, controls) -> np.ndarray:
    """The E8 decision vector: ``[states; controls]``, raveled."""
    return np.concatenate([states.ravel(), controls.ravel()])


def _trajectory_map(dynamics, horizon: int) -> np.ndarray:
    """The condensed ``Phi_x``: ``delta_x = Phi_x @ delta_u``.

    ``x_k = A^k x_0 + sum_{j<k} A^{k-1-j} B u_j``, so the coefficient of ``u_j``
    in ``x_k`` is ``A^{k-1-j} B`` -- the power advances with the *row* index.
    """
    a, b = dynamics.A, dynamics.B
    nx, nu = dynamics.state_dimension, dynamics.input_dimension
    blocks = np.zeros((horizon + 1, nx, horizon * nu))
    for step in range(horizon):
        power = np.eye(nx)
        for later in range(step + 1, horizon + 1):
            blocks[later, :, step * nu : (step + 1) * nu] = power @ b
            power = a @ power
    return blocks.reshape((horizon + 1) * nx, horizon * nu)


def _condensed_hessian(dynamics, horizon, weights, n_agents, phi_x) -> np.ndarray:
    """``H`` on the control sequence, per agent, tiled across agents."""
    nx = dynamics.state_dimension
    stage = np.block(
        [
            [weights.q_position, np.zeros_like(weights.q_position)],
            [np.zeros_like(weights.q_velocity), weights.q_velocity],
        ]
    )
    per_agent = np.zeros(((horizon + 1) * nx,) * 2)
    for step in range(horizon):
        start = step * nx
        per_agent[start : start + nx, start : start + nx] += stage
    per_agent[-nx:, -nx:] += weights.p_terminal
    state_cost = np.kron(np.eye(n_agents), per_agent)
    phi_all = np.kron(np.eye(n_agents), phi_x)
    r_block = np.kron(np.eye(n_agents * horizon), weights.r_input)
    return phi_all.T @ state_cost @ phi_all + r_block


def _edge_gradient(dynamics, horizon, offset_i, offset_j, states_i, states_j, metric):
    """Exact gradient of ``(1/2) sum_k ||y_i - y_j||^2_W`` w.r.t. both control sequences.

    ``y`` uses the formation-aligned output ``C x - d``, so the residual is
    ``C(x_i - x_j) - (d_i - d_j)`` and the ``j`` block of the gradient is the
    negation of the ``i`` block.
    """
    a, b, c = dynamics.A, dynamics.B, dynamics.C
    nx, nu = dynamics.state_dimension, dynamics.input_dimension
    residual = np.stack(
        [
            c @ states_i[step] - offset_i - (c @ states_j[step] - offset_j)
            for step in range(horizon + 1)
        ]
    )
    weighted = residual @ metric.T
    grad_i = np.zeros((horizon, nu))
    for m in range(horizon):
        adjoint = np.zeros(nx)
        for k in range(m + 1, horizon + 1):
            adjoint += np.linalg.matrix_power(a.T, k - 1 - m) @ c.T @ weighted[k]
        grad_i[m] = b.T @ adjoint
    return grad_i, -grad_i


def _perturbation_gradient(scenario, loc_states, phi_x):
    """``gamma * M(w) u*_loc`` summed over every edge, and the worst single edge."""
    dynamics = scenario.dynamics
    horizon = scenario.config.horizon
    nu = dynamics.input_dimension
    n_agents = scenario.graph.n_agents
    stride = horizon * nu
    total = np.zeros(n_agents * stride)
    worst_edge = 0.0
    for agent_i, agent_j in scenario.graph.edges:
        grad_i, grad_j = _edge_gradient(
            dynamics, horizon,
            scenario.graph.offsets[agent_i], scenario.graph.offsets[agent_j],
            loc_states[agent_i], loc_states[agent_j], scenario.weights.edge_metric,
        )
        edge = np.zeros_like(total)
        edge[agent_i * stride : (agent_i + 1) * stride] = grad_i.ravel()
        edge[agent_j * stride : (agent_j + 1) * stride] = grad_j.ravel()
        total += edge
        worst_edge = max(worst_edge, float(np.linalg.norm(edge)))
    return total, worst_edge


def _free_block(H, u_loc, config, horizon, nu, n_agents):
    """Boolean mask of the controls strictly inside the box at ``u*_loc``.

    A variable sitting exactly on its bound cannot move to first order, so the
    sensitivity formula must be solved on the complement of the active set.
    """
    lower = np.tile(config.acceleration_lower, n_agents * horizon)
    upper = np.tile(config.acceleration_upper, n_agents * horizon)
    active = (u_loc <= lower + 1e-9) | (u_loc >= upper - 1e-9)
    return ~active


def _run_seed(topology: str, seed: int) -> dict:
    scenario = build_scenario(seed, topology, horizon=HORIZON, rho=1.0)
    dynamics, weights, config = scenario.dynamics, scenario.weights, scenario.config
    horizon = config.horizon
    nu = dynamics.input_dimension
    n_agents = scenario.graph.n_agents
    n_edges = len(scenario.graph.edges)
    stride = horizon * nu

    loc = solve_centralized_qp(_uncoupled(scenario))
    loc_dec = _decision(loc.states, loc.controls)
    u_loc = loc.controls.ravel()

    phi_x = _trajectory_map(dynamics, horizon)
    phi_all = np.kron(np.eye(n_agents), phi_x)
    H = _condensed_hessian(dynamics, horizon, weights, n_agents, phi_x)
    gamma = float(weights.gamma)
    mu = float(np.min(np.linalg.eigvalsh(weights.r_input)))

    # --- Check 1: does the condensed Hessian respect the analytic modulus? ---
    lambda_min_condensed = float(np.min(np.linalg.eigvalsh(H)))

    # --- Check 2: unconditional bound, no active-set assumption ---------------
    grad_total, worst_edge = _perturbation_gradient(scenario, loc.states, phi_x)
    # The lift from ||du|| to ||[dx; du]||: sqrt(sigma_max(Phi_x)^2 + 1),
    # since the u block is orthogonal to the x block in the decision vector.
    lift = float(np.sqrt(np.linalg.norm(phi_x, 2) ** 2 + 1.0))
    bound_control = gamma * worst_edge / mu
    bound_decision = lift * bound_control

    # --- Check 3: sharp value, solved on the free block -----------------------
    free = _free_block(H, u_loc, config, horizon, nu, n_agents)
    h_ff = H[np.ix_(free, free)]
    grad_free = gamma * grad_total[free]
    step_free = np.linalg.solve(h_ff, grad_free)
    step = np.zeros_like(grad_total)
    step[free] = step_free
    sharp_decision = float(
        np.linalg.norm(np.concatenate([phi_all @ step, step])) / n_edges
    )
    sharp_control = float(np.linalg.norm(step) / n_edges)

    # --- Check 4: the submitted protocol's chord at s = 1, and the true limit --
    chord = 0.0
    for scale in E8_SCALES:
        sol = solve_centralized_qp(
            replace(
                scenario,
                graph=replace(
                    scenario.graph, edge_weights=scale * np.ones(n_edges)
                ),
            )
        )
        deviation = float(np.linalg.norm(_decision(sol.states, sol.controls) - loc_dec))
        chord = max(chord, deviation / (scale * n_edges))

    sol = solve_centralized_qp(
        replace(
            scenario,
            graph=replace(
                scenario.graph, edge_weights=SMALL_S * np.ones(n_edges)
            ),
        )
    )
    limit = float(
        np.linalg.norm(_decision(sol.states, sol.controls) - loc_dec)
        / (SMALL_S * n_edges)
    )

    return {
        "topology": topology,
        "seed": seed,
        "n_edges": n_edges,
        "n_active_bounds": int((~free).sum()),
        "mu_lambda_min_R": mu,
        "lambda_min_condensed": lambda_min_condensed,
        "modulus_claim_holds": bool(lambda_min_condensed >= mu - 1e-9),
        "L_bound_decision": bound_decision,
        "L_sharp_decision": sharp_decision,
        "L_chord_at_s1": chord,
        "L_limit_s_to_0": limit,
        "bound_holds": bool(bound_decision >= limit - 1e-9),
        "sharp_matches_limit": bool(
            abs(sharp_decision - limit) <= 1e-4 * max(1.0, limit)
        ),
        "chord_bias_factor": limit / chord if chord > 0 else float("inf"),
    }


def main() -> None:
    records = [_run_seed(topology, seed) for topology in TOPOLOGIES for seed in SEEDS]

    def _col(name):
        return [r[name] for r in records]

    summary = {
        "protocol": "E8 protocol: chain/ring/connected_random, 30 seeds, horizon 10",
        "mu": 0.15,
        "modulus_claim_holds_everywhere": all(_col("modulus_claim_holds")),
        "min_lambda_min_condensed": float(np.min(_col("lambda_min_condensed"))),
        "bound_holds_everywhere": all(_col("bound_holds")),
        "sharp_matches_limit_everywhere": all(_col("sharp_matches_limit")),
        "seeds_with_active_bounds": int(sum(r["n_active_bounds"] > 0 for r in records)),
        "L_bound_decision_mean": float(np.mean(_col("L_bound_decision"))),
        "L_sharp_decision_mean": float(np.mean(_col("L_sharp_decision"))),
        "L_chord_at_s1_mean": float(np.mean(_col("L_chord_at_s1"))),
        "L_chord_at_s1_std": float(np.std(_col("L_chord_at_s1"))),
        "L_limit_s_to_0_mean": float(np.mean(_col("L_limit_s_to_0"))),
        "L_limit_s_to_0_std": float(np.std(_col("L_limit_s_to_0"))),
        "L_limit_s_to_0_max": float(np.max(_col("L_limit_s_to_0"))),
        "mean_chord_bias_factor": float(np.mean(_col("chord_bias_factor"))),
        "max_chord_bias_factor": float(np.max(_col("chord_bias_factor"))),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
