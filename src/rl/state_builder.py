"""
EEQ401 — RL State Builder (77-Dimensional Observation Vector).

Constructs the RL observation from post-contingency PyPOWER solver
results, handling both convergent and non-convergent cases.

State layout (77 dims):
    [0:14]   Bus voltage magnitudes (V_bus1..V_bus14)
    [14:28]  Bus voltage angles (angle_bus1..angle_bus14)
    [28:48]  Line loading percentages (line_1..line_20)
    [48:53]  Generator active power outputs (gen_P1..gen_P5)
    [53:73]  One-hot line outage mask (20 dims, 1 = outaged)
    [73:77]  Derived summary: [V_min, V_max, L_max, total_load_mw]
"""

import numpy as np
from typing import Dict, Any, Optional

STATE_DIM = 77
NUM_BUSES = 14
NUM_LINES = 20
NUM_GENS = 5

# Sentinel values for non-convergent cases
NON_CONV_VOLTAGE = 0.0
NON_CONV_ANGLE = 0.0
NON_CONV_LOADING = 999.0
NON_CONV_GEN_POWER = 0.0


def build_state(
    results: Dict[str, Any],
    converged: bool,
    outaged_line_idx: Optional[int] = None,
    total_load_mw: Optional[float] = None,
) -> np.ndarray:
    """
    Constructs the 77-dimensional RL observation vector.

    Args:
        results (dict): Extracted results dict from extract_results().
        converged (bool): Whether AC power flow converged.
        outaged_line_idx (int, optional): Index of the outaged line (0-based).
            None means no outage (base case).
        total_load_mw (float, optional): Total system load in MW. Used as
            fallback for non-convergent cases where results are empty.

    Returns:
        state (np.ndarray): Shape (77,) float32 observation vector.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)

    if converged and results.get('convergence', False):
        # Bus voltages [0:14]
        bus_v = results.get('bus_voltages', [])
        for i in range(NUM_BUSES):
            state[i] = bus_v[i] if i < len(bus_v) else 1.0

        # Bus angles [14:28]
        bus_a = results.get('bus_angles', [])
        for i in range(NUM_BUSES):
            state[NUM_BUSES + i] = bus_a[i] if i < len(bus_a) else 0.0

        # Line loadings [28:48]
        line_l = results.get('line_loadings', [])
        for i in range(NUM_LINES):
            state[2 * NUM_BUSES + i] = line_l[i] if i < len(line_l) else 0.0

        # Generator powers [48:53]
        gen_p = results.get('gen_powers', [])
        for i in range(NUM_GENS):
            state[2 * NUM_BUSES + NUM_LINES + i] = gen_p[i] if i < len(gen_p) else 0.0

        # Derived summary [73:77]
        state[73] = results.get('min_voltage', 1.0)
        state[74] = results.get('max_voltage', 1.0)
        state[75] = results.get('max_line_loading', 0.0)
        state[76] = results.get('total_load_mw', total_load_mw or 0.0)

    else:
        # Non-convergent case: populate with sentinel values
        state[0:NUM_BUSES] = NON_CONV_VOLTAGE
        state[NUM_BUSES:2 * NUM_BUSES] = NON_CONV_ANGLE
        state[2 * NUM_BUSES:2 * NUM_BUSES + NUM_LINES] = NON_CONV_LOADING
        state[2 * NUM_BUSES + NUM_LINES:2 * NUM_BUSES + NUM_LINES + NUM_GENS] = NON_CONV_GEN_POWER

        # Summary features for non-convergent
        state[73] = 0.0   # V_min
        state[74] = 0.0   # V_max
        state[75] = 999.0  # L_max
        state[76] = total_load_mw if total_load_mw is not None else 0.0

    # One-hot outage mask [53:73]
    if outaged_line_idx is not None and 0 <= outaged_line_idx < NUM_LINES:
        state[2 * NUM_BUSES + NUM_LINES + NUM_GENS + outaged_line_idx] = 1.0

    return state
