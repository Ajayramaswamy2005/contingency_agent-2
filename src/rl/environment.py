"""
EEQ401 — Gymnasium Environment for IEEE 14-Bus Contingency Remediation.

Single-step MDP:
  reset()  -> Pick a scenario + line contingency -> Run AC PF -> Return post-contingency state S
  step(a)  -> Apply action a -> Re-run AC PF -> Return (S', reward, done=True, info)
"""

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Optional, Tuple

from src.network.load_network import load_network
from src.contingency.apply_contingency import apply_contingency
from src.simulation.run_load_flow import run_load_flow
from src.results.extract_results import extract_results
from src.labeling.classify_security import classify_security

from src.rl.state_builder import build_state, STATE_DIM
from src.rl.action_space import apply_action, NUM_ACTIONS, get_action_name
from src.rl.reward import compute_reward


class ContingencyRLEnv(gym.Env):
    """
    Single-step RL environment for N-1 contingency corrective action selection.
    Operates over a scenario pool.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        scenario_pool: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        scenario_indices: Optional[List[int]] = None,
        skip_base_cases: bool = True,
    ):
        super().__init__()

        self.cfg = cfg
        self.scenario_pool = scenario_pool
        self.skip_base_cases = skip_base_cases

        # Restrict to specified scenario indices
        if scenario_indices is not None:
            self.scenario_pool = [scenario_pool[i] for i in scenario_indices]

        # Load base mpc once
        self.base_mpc, self.meta = load_network()

        # Build episode index: list of (pool_idx, contingency_idx) pairs
        self._episode_index = []
        for pool_idx, sc in enumerate(self.scenario_pool):
            if not skip_base_cases:
                self._episode_index.append((pool_idx, None))  # base case
            for c_idx in range(len(sc['contingencies'])):
                self._episode_index.append((pool_idx, c_idx))

        # Gymnasium spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        # Current episode state (set in reset)
        self._current_mpc = None
        self._current_cont_mpc = None
        self._current_results = None
        self._current_converged = False
        self._current_label = 'Critical'
        self._current_outage_idx = None
        self._current_scenario_id = None
        self._current_cont_name = None
        self._total_load_mw = None

        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick random episode
        ep_idx = self._rng.integers(0, len(self._episode_index))
        pool_idx, cont_idx = self._episode_index[ep_idx]
        scenario = self.scenario_pool[pool_idx]

        self._current_scenario_id = scenario['scenario_id']
        load_pattern = scenario['load_pattern']

        # Build mpc with the scenario's load pattern applied
        self._current_mpc = copy.deepcopy(self.base_mpc)
        for bus_id, scale in load_pattern.items():
            mask = self._current_mpc['bus'][:, 0] == int(bus_id)
            self._current_mpc['bus'][mask, 2] *= float(scale)
            self._current_mpc['bus'][mask, 3] *= float(scale)

        self._total_load_mw = float(self._current_mpc['bus'][:, 2].sum())

        # Apply contingency
        if cont_idx is not None:
            contingency = scenario['contingencies'][cont_idx]
            self._current_outage_idx = contingency['element_idx']
            self._current_cont_name = contingency['element_name']
            self._current_cont_mpc = apply_contingency(self._current_mpc, contingency)
        else:
            self._current_outage_idx = None
            self._current_cont_name = 'Base Case'
            self._current_cont_mpc = copy.deepcopy(self._current_mpc)

        # Run AC power flow
        converged, raw_results = run_load_flow(mpc=self._current_cont_mpc)
        results = extract_results(raw_results, converged, meta=self.meta)
        label, violations = classify_security(results, converged, self.cfg)

        self._current_results = results
        self._current_converged = converged
        self._current_label = label

        obs = build_state(
            results, converged,
            outaged_line_idx=self._current_outage_idx,
            total_load_mw=self._total_load_mw,
        )

        info = {
            'scenario_id': self._current_scenario_id,
            'contingency': self._current_cont_name,
            'pre_label': label,
            'pre_converged': converged,
        }

        return obs, info

    def step(self, action: int):
        pre_label = self._current_label

        # Apply corrective action
        modified_mpc, action_cost = apply_action(self._current_cont_mpc, action)

        # Re-solve AC power flow
        converged, raw_results = run_load_flow(mpc=modified_mpc)
        results = extract_results(raw_results, converged, meta=self.meta)
        post_label, violations = classify_security(results, converged, self.cfg)

        obs = build_state(
            results, converged,
            outaged_line_idx=self._current_outage_idx,
            total_load_mw=self._total_load_mw,
        )

        reward, reward_breakdown = compute_reward(
            pre_label=pre_label,
            post_label=post_label,
            post_converged=converged,
            post_results=results,
            action_cost=action_cost,
            cfg=self.cfg,
        )

        info = {
            'scenario_id': self._current_scenario_id,
            'contingency': self._current_cont_name,
            'pre_label': pre_label,
            'post_label': post_label,
            'post_converged': converged,
            'action_name': get_action_name(action),
            'action_cost': action_cost,
            'reward_breakdown': reward_breakdown,
        }

        terminated = True
        truncated = False

        return obs, reward, terminated, truncated, info

    def get_num_episodes(self) -> int:
        return len(self._episode_index)
