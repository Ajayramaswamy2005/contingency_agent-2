"""
EEQ401 — Reward Function for RL Corrective Action Engine.

Evaluates the effectiveness of a corrective action by comparing
the resulting post-action security state against the original
post-contingency state.

Reward =  R_security
        - P_voltage
        - P_thermal
        - P_non_convergence
        - C_action
"""

from typing import Dict, Any, Tuple

# Reward component weights
REWARD_SAFE = 50.0
REWARD_ALERT = 10.0
BONUS_CRITICAL_TO_ALERT = 20.0
PENALTY_DEGRADATION_TO_CRITICAL = -30.0

PENALTY_PER_VOLTAGE_VIOLATION = 5.0
PENALTY_THERMAL_PER_PCT = 0.5  # per percentage point over 100%

PENALTY_NON_CONVERGENCE = 100.0


def compute_reward(
    pre_label: str,
    post_label: str,
    post_converged: bool,
    post_results: Dict[str, Any],
    action_cost: float,
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Computes the scalar reward for a corrective action.

    Args:
        pre_label (str): Security label BEFORE action ('Safe', 'Alert', 'Critical').
        post_label (str): Security label AFTER action.
        post_converged (bool): Whether AC power flow converged after action.
        post_results (dict): Extracted results dict after action.
        action_cost (float): Operational cost of the action.
        cfg (dict): Configuration dict with security thresholds.

    Returns:
        reward (float): Scalar reward.
        breakdown (dict): Reward component breakdown for diagnostics.
    """
    sec = cfg['security']
    v_safe_min = sec['v_safe_min']
    v_safe_max = sec['v_safe_max']
    line_safe_max = sec['line_safe_max']

    breakdown = {
        'R_security': 0.0,
        'P_voltage': 0.0,
        'P_thermal': 0.0,
        'P_non_conv': 0.0,
        'C_action': action_cost,
    }

    # 1. Non-convergence penalty
    if not post_converged:
        breakdown['P_non_conv'] = PENALTY_NON_CONVERGENCE
        reward = -PENALTY_NON_CONVERGENCE - action_cost
        breakdown['total'] = reward
        return reward, breakdown

    # 2. Security state transition reward
    if post_label == 'Safe':
        breakdown['R_security'] = REWARD_SAFE
    elif post_label == 'Alert':
        breakdown['R_security'] = REWARD_ALERT
        if pre_label == 'Critical':
            breakdown['R_security'] += BONUS_CRITICAL_TO_ALERT
    else:  # Critical
        if pre_label != 'Critical':
            # Action made things worse
            breakdown['R_security'] = PENALTY_DEGRADATION_TO_CRITICAL
        else:
            # Was already Critical, stayed Critical — no bonus
            breakdown['R_security'] = 0.0

    # 3. Voltage violation penalties
    bus_voltages = post_results.get('bus_voltages', [])
    n_v_violations = 0
    for v in bus_voltages:
        if v < v_safe_min or v > v_safe_max:
            n_v_violations += 1
    breakdown['P_voltage'] = PENALTY_PER_VOLTAGE_VIOLATION * n_v_violations

    # 4. Thermal overload penalties (penalty per % point above 100%)
    line_loadings = post_results.get('line_loadings', [])
    thermal_excess = 0.0
    for loading in line_loadings:
        if loading > line_safe_max:
            thermal_excess += (loading - line_safe_max)
    breakdown['P_thermal'] = PENALTY_THERMAL_PER_PCT * thermal_excess

    # 5. Final reward
    reward = (
        breakdown['R_security']
        - breakdown['P_voltage']
        - breakdown['P_thermal']
        - breakdown['C_action']
    )

    breakdown['total'] = reward
    return reward, breakdown
