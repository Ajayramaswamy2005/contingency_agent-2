"""
Unit tests for RL components and DQNAgent in Contingency-Analysis-2.
"""

import pytest
import numpy as np
import torch
from config.config import get_config
from src.rl.action_space import NUM_ACTIONS, get_action_name, apply_action
from src.rl.state_builder import build_state, STATE_DIM
from src.rl.reward import compute_reward
from src.rl.dqn_agent import DQNAgent, DQNNetwork, normalize_state
from pypower.api import case14

cfg = get_config()


class TestActionSpace:
    def test_action_count(self):
        assert NUM_ACTIONS == 20

    def test_do_nothing(self):
        mpc = case14()
        mod, cost = apply_action(mpc, 0)
        assert cost == 0.0
        np.testing.assert_array_equal(mod['bus'], mpc['bus'])

    def test_redispatch(self):
        mpc = case14()
        orig_pg = float(mpc['gen'][1, 1])
        mod, cost = apply_action(mpc, 1)
        assert float(mod['gen'][1, 1]) == pytest.approx(min(orig_pg + 10.0, 140.0))
        assert cost == 2.0


class TestStateBuilder:
    def test_state_shape(self):
        results = {
            'convergence': True,
            'bus_voltages': [1.0] * 14,
            'bus_angles': [0.0] * 14,
            'line_loadings': [50.0] * 20,
            'gen_powers': [100.0] * 5,
            'min_voltage': 0.95,
            'max_voltage': 1.06,
            'max_line_loading': 80.0,
            'total_load_mw': 259.0,
        }
        state = build_state(results, converged=True, outaged_line_idx=5)
        assert state.shape == (STATE_DIM,)
        assert state[53 + 5] == 1.0


class TestDQNAgent:
    def test_initialization(self):
        agent = DQNAgent(state_dim=77, num_actions=20, device='cpu')
        assert agent.state_dim == 77
        assert agent.num_actions == 20

    def test_action_selection(self):
        agent = DQNAgent(state_dim=77, num_actions=20, device='cpu')
        obs = np.ones(77, dtype=np.float32)
        act = agent.select_action(obs, training=False)
        assert 0 <= act < 20

    def test_train_step(self):
        agent = DQNAgent(state_dim=77, num_actions=20, batch_size=4, device='cpu')
        obs = np.ones(77, dtype=np.float32)
        for _ in range(8):
            agent.store_transition(obs, action=0, reward=1.0, next_obs=obs, done=False)
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
