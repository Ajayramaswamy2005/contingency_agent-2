"""
EEQ401 — 20-Action Corrective Action Space for IEEE 14-Bus Contingency Remediation.

Action Categories:
  A0:     Do Nothing (maintain current post-contingency state)
  A1-A8:  Generator Redispatch (+/- 10 MW at controllable non-slack generators)
  A9-A19: Controlled Load Shedding (10% curtailment at all candidate load buses)
"""

import copy
import numpy as np
from typing import Dict, Any, Tuple

NUM_ACTIONS = 20

# IEEE 14-bus generator specs: (gen_index_in_mpc, bus_id, Pmin, Pmax)
# Bus 1 is slack (auto-balances), not directly controllable as a corrective action
GEN_SPECS = [
    {'gen_idx': 1, 'bus_id': 2,  'Pmin': 0.0, 'Pmax': 140.0},
    {'gen_idx': 2, 'bus_id': 3,  'Pmin': 0.0, 'Pmax': 100.0},
    {'gen_idx': 3, 'bus_id': 6,  'Pmin': 0.0, 'Pmax': 100.0},
    {'gen_idx': 4, 'bus_id': 8,  'Pmin': 0.0, 'Pmax': 100.0},
]

# ALL load buses in IEEE 14-bus (every bus with Pd > 0 at nominal)
LOAD_BUSES = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]

# Cost parameters
REDISPATCH_COST = 2.0           # Flat cost per generator adjustment
LOAD_SHED_COST_PER_MW = 1.0    # Proportional cost per MW shed

# Shed percentage for load shedding actions
SHED_PCT = 0.10

# Action definitions: (category, description, parameters)
ACTION_DEFS = [
    # A0: Do Nothing
    ('do_nothing', 'Do Nothing', {}),

    # A1-A8: Generator Redispatch (4 gens x 2 directions)
    ('redispatch', 'Gen@Bus2 +10MW',  {'gen_idx': 1, 'bus_id': 2,  'delta_mw': +10.0, 'Pmin': 0.0, 'Pmax': 140.0}),
    ('redispatch', 'Gen@Bus2 -10MW',  {'gen_idx': 1, 'bus_id': 2,  'delta_mw': -10.0, 'Pmin': 0.0, 'Pmax': 140.0}),
    ('redispatch', 'Gen@Bus3 +10MW',  {'gen_idx': 2, 'bus_id': 3,  'delta_mw': +10.0, 'Pmin': 0.0, 'Pmax': 100.0}),
    ('redispatch', 'Gen@Bus3 -10MW',  {'gen_idx': 2, 'bus_id': 3,  'delta_mw': -10.0, 'Pmin': 0.0, 'Pmax': 100.0}),
    ('redispatch', 'Gen@Bus6 +10MW',  {'gen_idx': 3, 'bus_id': 6,  'delta_mw': +10.0, 'Pmin': 0.0, 'Pmax': 100.0}),
    ('redispatch', 'Gen@Bus6 -10MW',  {'gen_idx': 3, 'bus_id': 6,  'delta_mw': -10.0, 'Pmin': 0.0, 'Pmax': 100.0}),
    ('redispatch', 'Gen@Bus8 +10MW',  {'gen_idx': 4, 'bus_id': 8,  'delta_mw': +10.0, 'Pmin': 0.0, 'Pmax': 100.0}),
    ('redispatch', 'Gen@Bus8 -10MW',  {'gen_idx': 4, 'bus_id': 8,  'delta_mw': -10.0, 'Pmin': 0.0, 'Pmax': 100.0}),

    # A9-A19: Load Shedding at 10% for ALL 11 load buses
    ('load_shed', 'Shed10% Bus2',  {'bus_id': 2,  'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus3',  {'bus_id': 3,  'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus4',  {'bus_id': 4,  'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus5',  {'bus_id': 5,  'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus6',  {'bus_id': 6,  'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus9',  {'bus_id': 9,  'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus10', {'bus_id': 10, 'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus11', {'bus_id': 11, 'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus12', {'bus_id': 12, 'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus13', {'bus_id': 13, 'shed_pct': SHED_PCT}),
    ('load_shed', 'Shed10% Bus14', {'bus_id': 14, 'shed_pct': SHED_PCT}),
]

assert len(ACTION_DEFS) == NUM_ACTIONS, f"Expected {NUM_ACTIONS} actions, got {len(ACTION_DEFS)}"


def get_action_name(action_id: int) -> str:
    """Returns a human-readable name for the given action."""
    return ACTION_DEFS[action_id][1]


def apply_action(mpc: Dict[str, Any], action_id: int) -> Tuple[Dict[str, Any], float]:
    """
    Applies a corrective action to the PyPOWER mpc dictionary.

    Args:
        mpc (dict): PyPOWER case dict (should already have load pattern
                    and contingency applied). Will be deep-copied.
        action_id (int): Discrete action index in [0, NUM_ACTIONS-1].

    Returns:
        modified_mpc (dict): Deep-copied mpc with the action applied.
        action_cost (float): The operational cost of this action (>= 0).
    """
    if action_id < 0 or action_id >= NUM_ACTIONS:
        raise ValueError(f"Invalid action_id {action_id}. Must be in [0, {NUM_ACTIONS - 1}].")

    modified_mpc = copy.deepcopy(mpc)
    category, desc, params = ACTION_DEFS[action_id]

    if category == 'do_nothing':
        return modified_mpc, 0.0

    elif category == 'redispatch':
        gen_idx = params['gen_idx']
        delta_mw = params['delta_mw']
        p_min = params['Pmin']
        p_max = params['Pmax']

        current_pg = modified_mpc['gen'][gen_idx, 1]
        new_pg = current_pg + delta_mw

        # Clip to physical generator limits
        new_pg = float(np.clip(new_pg, p_min, p_max))

        modified_mpc['gen'][gen_idx, 1] = new_pg
        return modified_mpc, REDISPATCH_COST

    elif category == 'load_shed':
        bus_id = params['bus_id']
        shed_pct = params['shed_pct']

        bus_mask = modified_mpc['bus'][:, 0] == bus_id
        if not np.any(bus_mask):
            return modified_mpc, 0.0

        current_pd = modified_mpc['bus'][bus_mask, 2][0]
        mw_shed = current_pd * shed_pct

        # Proportional reduction to both P and Q (maintains power factor)
        modified_mpc['bus'][bus_mask, 2] *= (1.0 - shed_pct)
        modified_mpc['bus'][bus_mask, 3] *= (1.0 - shed_pct)

        # Cost proportional to actual MW shed
        cost = LOAD_SHED_COST_PER_MW * abs(mw_shed)
        return modified_mpc, cost

    else:
        raise ValueError(f"Unknown action category: {category}")
