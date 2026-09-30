"""
EEQ401 — MASTER TRAINING SCRIPT: DEEP Q-NETWORK (DQN) ON NEW 100k SAMPLES (GPU-ACCELERATED)

Trains a fresh DQN Agent from scratch on the new dataset samples alone (IEEE 14-bus, 20 N-1 contingencies per scenario).
Utilizes the high-memory GPU (48GB NVIDIA RTX A6000) for tensor acceleration and batch training.

Architecture:
  Input: 77-D continuous grid state (voltages, angles, loadings, generator MW, line outage mask, summaries)
  Policy Network: Linear(77 -> 256) -> LayerNorm -> ReLU -> Linear(256 -> 256) -> LayerNorm -> ReLU -> Linear(256 -> 20)
  Target Network: Identical architecture, periodically synchronized
  Action Space: 20 discrete physical actions (Do Nothing, 8 Generator Redispatch, 11 Load Shedding)
  Replay Buffer: 100,000 transition capacity
  Optimizer: Adam (lr=5e-4) with gradient clipping
  Loss: Huber / Mean Squared Bellman Error
"""

import sys
import os
import time
import json
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# Ensure local imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import get_config
from src.dataset.scenario_pool import load_scenario_pool
from src.rl.environment import ContingencyRLEnv
from src.rl.dqn_agent import DQNAgent, DQNNetwork, normalize_state
from src.rl.action_space import NUM_ACTIONS, get_action_name, apply_action
from src.rl.state_builder import build_state
from src.rl.reward import compute_reward
from src.network.load_network import load_network
from src.contingency.apply_contingency import apply_contingency
from src.simulation.run_load_flow import run_load_flow
from src.results.extract_results import extract_results
from src.labeling.classify_security import classify_security


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on New Samples using GPU")
    parser.add_argument("--pool-path", type=str, default="data/tabular/dataset_100k/dataset_100k_pool.json",
                        help="Path to new samples pool JSON")
    parser.add_argument("--episodes", type=int, default=25000,
                        help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size for GPU training")
    parser.add_argument("--buffer-capacity", type=int, default=100000,
                        help="Replay buffer capacity")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor")
    parser.add_argument("--train-scenarios", type=int, default=4000,
                        help="Number of scenarios to use for training (out of 5000)")
    parser.add_argument("--eval-scenarios", type=int, default=100,
                        help="Number of held-out test scenarios to evaluate after training")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    return parser.parse_args()


def print_gpu_info(device_str: str):
    print("=" * 70)
    print("  HARDWARE COMPUTE ENVIRONMENT")
    print("=" * 70)
    if torch.cuda.is_available() and "cuda" in device_str:
        dev_idx = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        total_mem_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(dev_idx) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(dev_idx) / (1024**3)
        print(f"  [+] Device:          CUDA:0 ({dev_name})")
        print(f"  [+] Total VRAM:      {total_mem_gb:.2f} GB")
        print(f"  [+] Currently Alloc: {allocated_gb:.2f} GB (Reserved: {reserved_gb:.2f} GB)")
        print(f"  [+] PyTorch Version: {torch.__version__}")
        print(f"  [+] CUDA Version:    {torch.version.cuda}")
    else:
        print(f"  [-] Device: CPU (CUDA available: {torch.cuda.is_available()})")
    print("=" * 70)


def main():
    args = parse_args()
    cfg = get_config()
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "█" * 70)
    print("  EEQ401 — DQN TRAINING FROM SCRATCH ON NEW SAMPLES (48GB GPU)")
    print("█" * 70)
    print_gpu_info(args.device)

    # -------------------------------------------------------------------------
    # STAGE 1: LOAD NEW SAMPLES SCENARIO POOL
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 1/4] Loading New Scenario Pool from: {args.pool_path}")
    t0 = time.time()
    if not os.path.exists(args.pool_path):
        raise FileNotFoundError(f"Scenario pool file not found: {args.pool_path}")

    pool = load_scenario_pool(args.pool_path)
    total_scenarios = len(pool)
    cases_per_sc = len(pool[0]['contingencies'])
    total_cases = total_scenarios * cases_per_sc
    print(f"  [+] Successfully loaded {total_scenarios} scenarios ({total_cases:,} total contingency cases) in {time.time() - t0:.2f}s")

    # Partition scenarios into Train and Test
    num_train = min(args.train_scenarios, total_scenarios)
    train_indices = list(range(0, num_train))
    test_indices = list(range(num_train, min(num_train + args.eval_scenarios, total_scenarios)))

    print(f"  [+] Train Scenarios:  {len(train_indices):,} ({len(train_indices)*cases_per_sc:,} cases)")
    print(f"  [+] Test Scenarios:   {len(test_indices):,} ({len(test_indices)*cases_per_sc:,} cases)")

    # -------------------------------------------------------------------------
    # STAGE 2: INITIALIZE DQN AGENT FROM SCRATCH
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 2/4] Initializing Fresh Deep Q-Network Agent on {args.device}...")
    agent = DQNAgent(
        state_dim=77,
        num_actions=NUM_ACTIONS,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=int(args.episodes * 0.6),
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=200,
        device=args.device,
    )

    env = ContingencyRLEnv(
        scenario_pool=pool,
        cfg=cfg,
        scenario_indices=train_indices,
        skip_base_cases=True,
    )

    # -------------------------------------------------------------------------
    # STAGE 3: TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 3/4] Training for {args.episodes:,} Episodes (Batch Size={args.batch_size})...")
    t_train_start = time.time()

    episode_rewards = []
    episode_labels = []
    recent_losses = []
    history_log = []

    pbar = tqdm(range(args.episodes), desc="  DQN Training (GPU)", unit="ep")

    for ep in pbar:
        obs, info = env.reset()
        action = agent.select_action(obs, info, training=True)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        agent.store_transition(obs, action, reward, next_obs, terminated)
        loss = agent.train_step()
        agent.update_epsilon(ep)

        episode_rewards.append(reward)
        post_label = step_info['post_label']
        episode_labels.append(post_label)
        if loss is not None:
            recent_losses.append(loss)

        # Log and progress updates
        if (ep + 1) % 50 == 0 or ep == args.episodes - 1:
            recent_w = episode_rewards[-100:]
            recent_safe = sum(1 for l in episode_labels[-100:] if l == 'Safe')
            avg_loss = float(np.mean(recent_losses[-50:])) if recent_losses else 0.0

            pbar.set_postfix({
                'eps': f"{agent.epsilon:.2f}",
                'AvgRew': f"{np.mean(recent_w):.1f}",
                'Safe%': f"{100*recent_safe/len(recent_w):.0f}%",
                'Loss': f"{avg_loss:.3f}",
                'Mem': f"{len(agent.memory):,}",
            })

            history_log.append({
                'episode': ep + 1,
                'epsilon': round(float(agent.epsilon), 4),
                'avg_reward_100': round(float(np.mean(recent_w)), 2),
                'safe_rate_100': round(100.0 * recent_safe / len(recent_w), 1),
                'loss': round(avg_loss, 4),
                'buffer_size': len(agent.memory),
                'elapsed_sec': round(time.time() - t_train_start, 1),
            })

    total_train_time = time.time() - t_train_start
    print(f"\n  [+] Training Complete in {total_train_time:.1f}s ({total_train_time/60:.2f} mins)!")

    # Save model weights and training logs
    model_weights_path = os.path.join(args.save_dir, 'dqn_agent_new_samples.pt')
    agent.save(model_weights_path)
    print(f"  [+] Saved PyTorch Model Weights to: {model_weights_path}")

    history_csv_path = os.path.join(args.save_dir, 'training_history.csv')
    pd.DataFrame(history_log).to_csv(history_csv_path, index=False)
    print(f"  [+] Saved Training History CSV to:   {history_csv_path}")

    # -------------------------------------------------------------------------
    # STAGE 4: BENCHMARK EVALUATION ON HELD-OUT TEST SCENARIOS
    # -------------------------------------------------------------------------
    if test_indices:
        print(f"\n[STAGE 4/4] 3-Way Benchmark Evaluation on {len(test_indices)} Held-Out Test Scenarios...")
        base_mpc, meta = load_network()
        t_eval_start = time.time()

        total_test = 0
        needs_action_count = 0
        no_action_safe = 0
        rl_safe = 0
        offline_safe = 0
        rl_matches_best = 0
        rl_matches_top3 = 0
        total_rl_load_shed = 0.0
        total_offline_load_shed = 0.0
        eval_cases = []

        eval_pbar = tqdm(test_indices, desc="  Evaluation", unit="sc")
        for sc_idx in eval_pbar:
            scenario = pool[sc_idx]
            sc_id = scenario['scenario_id']
            load_pattern = scenario['load_pattern']

            loaded_mpc = copy.deepcopy(base_mpc)
            for bus_id, scale in load_pattern.items():
                mask = loaded_mpc['bus'][:, 0] == int(bus_id)
                loaded_mpc['bus'][mask, 2] *= float(scale)
                loaded_mpc['bus'][mask, 3] *= float(scale)

            total_load_mw = float(loaded_mpc['bus'][:, 2].sum())

            for cont in scenario['contingencies']:
                total_test += 1
                cont_name = cont['element_name']
                outage_idx = cont['element_idx']

                cont_mpc = apply_contingency(loaded_mpc, cont)

                # 1. No Action baseline
                conv_pre, raw_pre = run_load_flow(mpc=cont_mpc)
                res_pre = extract_results(raw_pre, conv_pre, meta=meta)
                label_pre, _ = classify_security(res_pre, conv_pre, cfg)

                if label_pre != 'Safe':
                    needs_action_count += 1
                if label_pre == 'Safe':
                    no_action_safe += 1

                # 2. Trained DQN Policy action
                obs_pre = build_state(res_pre, conv_pre, outaged_line_idx=outage_idx, total_load_mw=total_load_mw)
                info_pre = {'scenario_id': sc_id, 'contingency': cont_name, 'pre_label': label_pre, 'pre_converged': conv_pre}
                rl_action = agent.select_action(obs_pre, info_pre, training=False)

                rl_mpc, rl_cost = apply_action(cont_mpc, rl_action)
                rl_conv, rl_raw = run_load_flow(mpc=rl_mpc)
                rl_res = extract_results(rl_raw, rl_conv, meta=meta)
                rl_label, _ = classify_security(rl_res, rl_conv, cfg)
                rl_reward, _ = compute_reward(label_pre, rl_label, rl_conv, rl_res, rl_cost, cfg)

                if rl_label == 'Safe':
                    rl_safe += 1
                if rl_action >= 9:
                    total_rl_load_shed += rl_cost

                # 3. Offline Exhaustive Best (Oracle)
                action_rewards = []
                for a in range(NUM_ACTIONS):
                    a_mpc, a_cost = apply_action(cont_mpc, a)
                    a_conv, a_raw = run_load_flow(mpc=a_mpc)
                    a_res = extract_results(a_raw, a_conv, meta=meta)
                    a_label, _ = classify_security(a_res, a_conv, cfg)
                    a_reward, _ = compute_reward(label_pre, a_label, a_conv, a_res, a_cost, cfg)
                    action_rewards.append((a, float(a_reward), a_label, a_cost))

                action_rewards.sort(key=lambda x: x[1], reverse=True)
                best_action, best_reward, best_label, best_cost = action_rewards[0]
                top3 = [x[0] for x in action_rewards[:3]]

                if best_label == 'Safe':
                    offline_safe += 1
                if rl_action == best_action:
                    rl_matches_best += 1
                if rl_action in top3:
                    rl_matches_top3 += 1
                if best_action >= 9:
                    total_offline_load_shed += best_cost

                eval_cases.append({
                    'scenario_id': sc_id,
                    'contingency': cont_name,
                    'pre_label': label_pre,
                    'rl_action': get_action_name(rl_action),
                    'rl_label': rl_label,
                    'rl_reward': round(float(rl_reward), 2),
                    'offline_best_action': get_action_name(best_action),
                    'offline_best_label': best_label,
                    'top1_match': (rl_action == best_action),
                    'top3_match': (rl_action in top3),
                })

            eval_pbar.set_postfix({
                'RLSafe': f"{100*rl_safe/total_test:.1f}%",
                'Top1': f"{100*rl_matches_best/total_test:.1f}%",
                'Top3': f"{100*rl_matches_top3/total_test:.1f}%",
            })

        total_eval_time = time.time() - t_eval_start

        metrics = {
            'total_test_cases': total_test,
            'insecure_cases_needing_mitigation': needs_action_count,
            'no_action_safe_pct': round(100.0 * no_action_safe / total_test, 1),
            'rl_agent_safe_pct': round(100.0 * rl_safe / total_test, 1),
            'offline_best_safe_pct': round(100.0 * offline_safe / total_test, 1),
            'rl_mitigation_success_pct': round(100.0 * (rl_safe - no_action_safe) / max(1, needs_action_count), 1),
            'policy_alignment_top1_pct': round(100.0 * rl_matches_best / total_test, 1),
            'policy_alignment_top3_pct': round(100.0 * rl_matches_top3 / total_test, 1),
            'avg_rl_load_shed_mw': round(total_rl_load_shed / total_test, 3),
            'avg_offline_load_shed_mw': round(total_offline_load_shed / total_test, 3),
            'train_time_seconds': round(total_train_time, 1),
            'eval_time_seconds': round(total_eval_time, 1),
        }

        summary_json_path = os.path.join(args.save_dir, 'evaluation_summary.json')
        with open(summary_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        cases_csv_path = os.path.join(args.save_dir, 'evaluation_cases.csv')
        pd.DataFrame(eval_cases).to_csv(cases_csv_path, index=False)

        print("\n" + "═" * 70)
        print("  FINAL EVALUATION REPORT — NEW SAMPLES DQN POLICY")
        print("═" * 70)
        print(f"  Test Cases Evaluated:                {total_test}")
        print(f"  Insecure Contingencies Needing Help: {needs_action_count}")
        print(f"  1. No-Action Security Rate:          {metrics['no_action_safe_pct']}% ({no_action_safe}/{total_test})")
        print(f"  2. DQN Agent Security Rate:          {metrics['rl_agent_safe_pct']}% ({rl_safe}/{total_test})")
        print(f"  3. Offline Best (Oracle) Rate:       {metrics['offline_best_safe_pct']}% ({offline_safe}/{total_test})")
        print(f"  -------------------------------------------------------------")
        print(f"  Top-1 Policy Alignment Accuracy:     {metrics['policy_alignment_top1_pct']}%")
        print(f"  Top-3 Policy Alignment Accuracy:     {metrics['policy_alignment_top3_pct']}%")
        print(f"  Avg RL Load Shed:                    {metrics['avg_rl_load_shed_mw']} MW/case")
        print("═" * 70)
        print(f"  [+] Saved evaluation summary: {summary_json_path}")
        print(f"  [+] Saved case breakdown:     {cases_csv_path}")
        print("═" * 70 + "\n")


if __name__ == '__main__':
    main()
