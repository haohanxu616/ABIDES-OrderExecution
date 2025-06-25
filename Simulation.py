import time
import numpy as np
from typing import Tuple, List, Dict
import logging

from BrokersEnv import SubGymMarketsExecutionEnv_v0
from QLearningAgent import QLearningAgent

logger = logging.getLogger(__name__)

def run_simulation(strategy: str, num_runs: int = 10, train_episodes: int = 10, eval_episodes: int = 10) -> Tuple[List[float], List[float], List[List[Tuple[float, float]]], List[List[float]]]:
    """Run a simulation for a given strategy with multiple runs and split training/evaluation phases.

    Parameters
    ----------
    strategy : str
        The execution strategy (TWAP, VWAP, IS, or POV).
    num_runs : int, optional
        Number of runs with different random seeds (default: 10).
    train_episodes : int, optional
        Number of training episodes per run (default: 10).
    eval_episodes : int, optional
        Number of evaluation episodes per run (default: 10).

    Returns
    -------
    tuple
        (rewards, execution_costs, trade_schedules, market_volumes) aggregated across runs.

    Notes
    -----
    Executes the Q-learning agent with a 300s timeout per run, aggregating results.
    """
    all_rewards = []
    all_execution_costs = []
    all_trade_schedules = []
    all_market_volumes = []

    for run in range(num_runs):
        logger.info(f"Starting run {run + 1}/{num_runs} for strategy {strategy}")
        env = SubGymMarketsExecutionEnv_v0(parent_order_size=500, max_order_size=100)
        env.strategy = strategy
        agent = QLearningAgent(env.action_space, strategy)
        rewards = []
        execution_costs = []
        trade_schedules = []
        market_volumes = []
        start_time = time.time()
        timeout = 300

        # Training phase
        for episode in range(train_episodes):
            state = env.reset()
            if state is None or time.time() - start_time > timeout:
                logger.error(f"Timeout or invalid reset after {time.time() - start_time:.1f} seconds in training")
                break
            done = False
            episode_reward = 0.0
            env.custom_metrics_tracker.execution_cost = 0.0
            env.custom_metrics_tracker.trade_schedule = []
            episode_volumes = []
            while not done and time.time() - start_time < timeout:
                action = agent.choose_action(state.flatten(), env)
                next_state, reward, done, info = env.step(action)
                if next_state is None:
                    logger.error("Invalid next_state received")
                    break
                agent.learn(state.flatten(), action, reward, next_state.flatten(), env)
                episode_reward += reward
                episode_volumes.append(info["volume"])
                state = next_state
            rewards.append(episode_reward)
            execution_costs.append(info["execution_cost"])
            trade_schedules.append(info["trade_schedule"])
            market_volumes.append(episode_volumes)
            if episode % 3 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Training Episode {episode}, Reward: {np.mean(rewards[-2:]):.4f}, Cost: {np.mean(execution_costs[-2:]):.4f}, Time: {elapsed:.1f}s")

        # Evaluation phase (no learning)
        for episode in range(eval_episodes):
            state = env.reset()
            if state is None or time.time() - start_time > timeout:
                logger.error(f"Timeout or invalid reset after {time.time() - start_time:.1f} seconds in evaluation")
                break
            done = False
            episode_reward = 0.0
            env.custom_metrics_tracker.execution_cost = 0.0
            env.custom_metrics_tracker.trade_schedule = []
            episode_volumes = []
            while not done and time.time() - start_time < timeout:
                action = agent.choose_action(state.flatten(), env)
                next_state, reward, done, info = env.step(action)
                if next_state is None:
                    logger.error("Invalid next_state received")
                    break
                episode_reward += reward
                episode_volumes.append(info["volume"])
                state = next_state
            rewards.append(episode_reward)
            execution_costs.append(info["execution_cost"])
            trade_schedules.append(info["trade_schedule"])
            market_volumes.append(episode_volumes)
            if episode % 3 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Evaluation Episode {episode}, Reward: {np.mean(rewards[-2:]):.4f}, Cost: {np.mean(execution_costs[-2:]):.4f}, Time: {elapsed:.1f}s")

        all_rewards.append(rewards)
        all_execution_costs.append(execution_costs)
        all_trade_schedules.append(trade_schedules)
        all_market_volumes.append(market_volumes)

    # Aggregate results across runs
    aggregated_rewards = [r for run in all_rewards for r in run]
    aggregated_costs = [c for run in all_execution_costs for c in run]
    aggregated_schedules = [s for run in all_trade_schedules for s in run]
    aggregated_volumes = [v for run in all_market_volumes for v in run]
    return aggregated_rewards, aggregated_costs, aggregated_schedules, aggregated_volumes

def compute_theoretical_trajectory(strategy: str, parent_order_size: int = 500, steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Compute theoretical cumulative executed quantities for a strategy.

    Parameters
    ----------
    strategy : str
        The execution strategy (TWAP, VWAP, IS, or POV).
    parent_order_size : int
        The total order size (default: 500).
    steps : int, optional
        Number of timesteps to compute trajectory (default: 12).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (times, quantities) arrays representing the theoretical trajectory.

    Notes
    -----
    Ensures quantities are non-negative with a single step count.
    """
    times = np.linspace(0, 1, steps)
    if strategy == "TWAP":
        quantities = times * parent_order_size
    elif strategy == "VWAP":
        step_weights = np.linspace(0, 1, steps) ** 2
        quantities = step_weights * parent_order_size
    elif strategy == "IS":
        quantities = np.minimum(times * parent_order_size * 0.3, np.linspace(0, parent_order_size, steps))
    elif strategy == "POV":
        base_volume_factor = 0.1
        quantities = np.minimum(times * parent_order_size * base_volume_factor * 10, times * parent_order_size * np.sqrt(times))
    quantities = np.maximum(0, quantities)  # Prevent negative quantities
    logger.debug(f"{strategy} Theoretical Trajectory: times={times}, quantities={quantities}")
    return times, quantities