import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import os

from Simulation import run_simulation, compute_theoretical_trajectory

logger = logging.getLogger(__name__)

def main():
    """Run simulations for all strategies, save results, and generate plots."""
    # Define the list of strategies to test
    strategies = ["TWAP", "VWAP", "IS", "POV"]
    # Initialize dictionary to store results
    all_results = {}
    # Iterate over each strategy
    for strategy in strategies:
        # Print the start of training for the current strategy
        print(f"Training {strategy}...")
        # Run simulation with multiple runs and split phases
        all_results[strategy] = run_simulation(strategy, num_runs=10, train_episodes=10, eval_episodes=10)

    # Import matplotlib for plotting with Agg backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    # Create directory if it doesn't exist
    output_dir = "/gpfs/home/haohxu/ABIDES-results/Experiment1/"
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure for reward curves
    plt.figure(figsize=(10, 6))
    # Plot reward curves for each strategy, aggregated across runs
    for strategy, (rewards, _, _, _) in all_results.items():
        plt.plot(np.arange(len(rewards)), rewards, label=strategy)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Curves (Aggregated Across 10 Runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward_curves.png"))
    plt.close()

    # Create a figure for execution cost comparison
    plt.figure(figsize=(10, 6))
    costs = [np.mean(all_results[strategy][1]) for strategy in strategies]  # Mean cost across runs
    plt.boxplot([all_results[strategy][1] for strategy in strategies], labels=strategies)
    plt.xlabel('Strategy')
    plt.ylabel('Execution Cost (Normalized)')
    plt.title('Execution Cost Comparison (Across 10 Runs)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "execution_cost_comparison.png"))
    plt.close()

    # Generate four separate plots for each strategy with single theoretical trajectory
    for strategy in strategies:
        rewards, execution_costs, trade_schedules, market_volumes = all_results[strategy]
        plt.figure(figsize=(10, 6))
        # Plot learned schedule from the last run
        schedule = trade_schedules[-1] if trade_schedules else []
        times, quantities = zip(*schedule) if schedule else ([], [])
        if times:
            plt.plot(times, quantities, 'b-', label=f"{strategy} Learned Trades")
            logger.debug(f"{strategy} Learned Trades: times={times}, quantities={quantities}")
        # Plot single theoretical schedule
        theo_times, theo_quantities = compute_theoretical_trajectory(strategy)
        plt.plot(theo_times, theo_quantities, 'r--', label=f"{strategy} Theoretical Trades")
        logger.debug(f"{strategy} Theoretical Trades: times={theo_times}, quantities={theo_quantities}")
        plt.xlabel('Time (Fraction of Execution Window)')
        plt.ylabel('Cumulative Executed Quantity')
        plt.title(f'Trade Schedule for {strategy} (10 Runs)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"trade_schedule_{strategy}.png"))
        plt.close()

if __name__ == "__main__":
    main()