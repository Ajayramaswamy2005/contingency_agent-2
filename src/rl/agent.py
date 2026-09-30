"""
EEQ401 — Tabular Q-Learning Agent for Contingency Corrective Actions.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict


def _severity_to_bucket(label: str, converged: bool) -> int:
    if not converged:
        return 0
    if label == 'Critical':
        return 1
    if label == 'Alert':
        return 2
    return 3


class QLearningAgent:
    def __init__(
        self,
        num_actions: int = 20,
        alpha: float = 0.1,
        gamma: float = 0.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions)
        )
        self.step_count = 0

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _make_key(self, info: Dict[str, Any]) -> Tuple:
        scenario_id = info.get('scenario_id', 'unknown')
        contingency = info.get('contingency', 'unknown')
        label = info.get('pre_label', 'Critical')
        converged = info.get('pre_converged', False)
        bucket = _severity_to_bucket(label, converged)
        return (scenario_id, contingency, bucket)

    def select_action(self, obs: np.ndarray, info: Dict[str, Any], training: bool = True) -> int:
        key = self._make_key(info)
        q_values = self.q_table[key]

        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(np.random.choice(best_actions))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Dict[str, Any],
    ) -> float:
        key = self._make_key(info)
        old_q = self.q_table[key][action]

        if done or self.gamma == 0.0:
            target = reward
        else:
            next_key = self._make_key(info)
            target = reward + self.gamma * np.max(self.q_table[next_key])

        td_error = target - old_q
        self.q_table[key][action] = old_q + self.alpha * td_error
        self.step_count += 1
        return td_error
