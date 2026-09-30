"""
EEQ401 — Deep Q-Network (DQN) Agent for Grid Contingency Remediation.

Features:
  - 77-dimensional continuous state representation
  - Normalized input state mapping (voltages, angles, loadings, gen MW, outage mask)
  - PyTorch neural network policy with LayerNorm and ReLU activations
  - Experience Replay Memory with minibatched gradient descent
  - Stable Bellman target calculation via dedicated Target Network
  - Supports GPU / CUDA execution for rapid learning
"""

import os
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def normalize_state(state: np.ndarray) -> np.ndarray:
    """
    Normalizes the 77-dimensional observation vector for neural network stability.
        [0:14]   Bus voltages -> centered near 1.0 (already standard p.u.)
        [14:28]  Bus angles -> scale / 30.0 (degrees to near [-1, 1])
        [28:48]  Line loading % -> scale / 100.0, clip at 3.0
        [48:53]  Generator MW outputs -> scale / 100.0
        [53:73]  One-hot line outage mask (0 or 1)
        [73:77]  Summary: [V_min, V_max, L_max, total_load_mw]
    """
    norm_s = np.array(state, dtype=np.float32).copy()

    # Bus angles: scale by 30.0
    norm_s[14:28] = norm_s[14:28] / 30.0

    # Line loadings: scale by 100.0, clip at 3.0
    norm_s[28:48] = np.clip(norm_s[28:48] / 100.0, 0.0, 3.0)

    # Generator powers: scale by 100.0
    norm_s[48:53] = norm_s[48:53] / 100.0

    # Summary features:
    norm_s[75] = np.clip(norm_s[75] / 100.0, 0.0, 3.0)  # L_max
    norm_s[76] = norm_s[76] / 300.0                     # Total load MW

    return norm_s


class DQNNetwork(nn.Module):
    """
    Deep Q-Network mapping 77-dim continuous grid states to Q-values for 20 discrete actions.
    Architecture: Linear(77 -> 256) -> LayerNorm -> ReLU -> Linear(256 -> 256) -> LayerNorm -> ReLU -> Linear(256 -> 20)
    """
    def __init__(self, state_dim: int = 77, action_dim: int = 20, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for storing transitions (s, a, r, s', done).
    """
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent managing policy network, target network, replay memory, and Bellman updates.
    """
    def __init__(
        self,
        state_dim: int = 77,
        num_actions: int = 20,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 12000,
        buffer_capacity: int = 100000,
        batch_size: int = 128,
        target_update_freq: int = 200,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Device selection: auto-detect GPU/CUDA
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"  [DQNAgent] Initialized on device: {self.device}")
        if self.device.type == "cuda":
            print(f"  [DQNAgent] GPU: {torch.cuda.get_device_name(self.device)}")

        # Neural Networks
        self.policy_net = DQNNetwork(state_dim, num_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.loss_fn = nn.MSELoss()

    def select_action(self, obs: np.ndarray, info: Optional[Dict] = None, training: bool = True) -> int:
        """
        Selects action using epsilon-greedy strategy during training or greedy policy during eval.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        norm_s = normalize_state(obs)
        state_t = torch.tensor(norm_s, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action = int(q_values.argmax(dim=1).item())

        if training:
            self.policy_net.train()

        return action

    def store_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        """
        Pushes normalized transition to replay buffer.
        """
        norm_s = normalize_state(obs)
        norm_ns = normalize_state(next_obs)
        self.memory.push(norm_s, action, reward, norm_ns, done)

    def train_step(self) -> Optional[float]:
        """
        Performs one gradient descent step over a mini-batch sampled from replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q values: Q(s, a)
        self.policy_net.train()
        q_eval = self.policy_net(states_t).gather(1, actions_t)

        # Target Q values: r + gamma * max_a Q_target(s', a) * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            q_target = rewards_t + (self.gamma * q_next * (1.0 - dones_t))

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1

        # Periodically update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def update_epsilon(self, current_step: int):
        """
        Linear decay of epsilon from epsilon_start to epsilon_end over epsilon_decay_steps.
        """
        fraction = min(1.0, float(current_step) / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def save(self, file_path: str):
        """Saves model weights to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, file_path)

    def load(self, file_path: str):
        """Loads model weights from disk."""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.step_count = checkpoint.get('step_count', 0)
        self.policy_net.eval()
        self.target_net.eval()
