import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple
from abc import ABC

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator
from abides_gym.envs.markets_environment import AbidesGymMarketsEnv
import logging

# Configure logging with debug level and ensure a handler exists
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
logger.handlers[0].setFormatter(logging.Formatter('{asctime} - {levelname} - {message}', style='{'))

class SubGymMarketsExecutionEnv_v0(AbidesGymMarketsEnv):
    """A Gym environment for algorithmic order execution using ABIDES.

    Parameters
    ----------
    background_config : Any, optional
        The configuration module for background agents (default: "rmsc04").
    mkt_close : str, optional
        Market close time (default: "09:40:00").
    timestep_duration : str, optional
        Duration between agent wakeups (default: "5s").
    starting_cash : int, optional
        Initial cash for the agent (default: 10000).
    max_order_size : int, optional
        Maximum order size per trade (default: 100).
    state_history_length : int, optional
        Length of state history buffer (default: 10, increased for robustness).
    market_data_buffer_length : int, optional
        Length of market data buffer (default: 10, increased for robustness).
    first_interval : str, optional
        Initial delay before first wakeup (default: "00:00:30", increased for warmup).
    parent_order_size : int, optional
        Total size of the parent order (default: 500).
    execution_window : str, optional
        Time window for order execution (default: "00:01:30").
    direction : str, optional
        Trade direction (default: "BUY").
    not_enough_reward_update : int, optional
        Penalty for under-execution (default: -1000).
    too_much_reward_update : int, optional
        Penalty for over-execution (default: -100).
    just_quantity_reward_update : int, optional
        Reward for exact execution (default: 0).
    debug_mode : bool, optional
        Enable debug logging (default: True).
    background_config_extra_kvargs : Dict[str, Any], optional
        Extra arguments for background config (default: {"num_noise_agents": 5}).

    Attributes
    ----------
    custom_metrics_tracker : CustomMetricsTracker
        Tracks metrics like rewards, quantities, and trade schedule.
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = markets_agent_utils.ignore_mkt_data_buffer_decorator

    @dataclass
    class CustomMetricsTracker(ABC):
        """Tracks custom metrics for the environment.

        Attributes
        ----------
        slippage_reward : float
            Reward from slippage (default: 0.0).
        late_penalty_reward : float
            Penalty for late execution (default: 0.0).
        executed_quantity : int
            Total executed quantity (default: 0).
        remaining_quantity : int
            Remaining quantity to execute (default: 0).
        action_counter : Dict[str, int]
            Counter for each action type (default: empty dict).
        holdings_pct : float
            Percentage of holdings (default: 0.0).
        time_pct : float
            Percentage of execution time elapsed (default: 0.0).
        diff_pct : float
            Difference between holdings and time percentage (default: 0.0).
        imbalance_all : float
            Market imbalance at all depths (default: 0.0).
        imbalance_5 : float
            Market imbalance at 5 levels (default: 0.0).
        price_impact : float
            Price impact of trades (default: 0.0).
        spread : float
            Bid-ask spread (default: 0.0).
        direction_feature : float
            Direction feature (default: 0.0).
        num_max_steps_per_episode : float
            Maximum steps per episode (default: 0.0).
        trade_schedule : List[Tuple[float, float]]
            Schedule of (time_pct, executed_quantity) pairs (default: empty list).
        """
        slippage_reward: float = 0.0
        late_penalty_reward: float = 0.0
        executed_quantity: int = 0
        remaining_quantity: int = 0
        action_counter: Dict[str, int] = field(default_factory=dict)
        holdings_pct: float = 0.0
        time_pct: float = 0.0
        diff_pct: float = 0.0
        imbalance_all: float = 0.0
        imbalance_5: float = 0.0
        price_impact: float = 0.0
        spread: float = 0.0
        direction_feature: float = 0.0
        num_max_steps_per_episode: float = 0.0
        trade_schedule: List[Tuple[float, float]] = field(default_factory=list)
        volume: float = 0.0
        execution_cost: float = 0.0

    def __init__(
        self,
        background_config: Any = "rmsc04",
        mkt_close: str = "09:40:00",
        timestep_duration: str = "5s",
        starting_cash: int = 10000,
        max_order_size: int = 100,
        state_history_length: int = 10,
        market_data_buffer_length: int = 10,
        first_interval: str = "00:00:30",
        parent_order_size: int = 500,
        execution_window: str = "00:01:30",
        direction: str = "BUY",
        not_enough_reward_update: int = -1000,
        too_much_reward_update: int = -100,
        just_quantity_reward_update: int = 0,
        debug_mode: bool = True,
        background_config_extra_kvargs: Dict[str, Any] = {"num_noise_agents": 5},
    ) -> None:
        """Initialize the environment with rigorous parameter validation."""
        self.background_config = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None)
        self.mkt_close = str_to_ns(mkt_close)
        self.timestep_duration = str_to_ns(timestep_duration)
        self.starting_cash = starting_cash
        self.max_order_size = max_order_size
        self.state_history_length = state_history_length
        self.market_data_buffer_length = market_data_buffer_length
        self.first_interval = str_to_ns(first_interval)
        self.parent_order_size = parent_order_size
        self.execution_window = str_to_ns(execution_window)
        self.direction = direction
        self.debug_mode = debug_mode
        self.too_much_reward_update = too_much_reward_update
        self.not_enough_reward_update = not_enough_reward_update
        self.just_quantity_reward_update = just_quantity_reward_update
        self.entry_price = 0.0
        self.near_touch = 0.0
        self.step_index = 0
        self.custom_metrics_tracker = self.CustomMetricsTracker()

        # Rigorous parameter validation
        assert background_config in ["rmsc03", "rmsc04", "smc_01"], "Invalid background config"
        assert self.first_interval >= 0 and self.first_interval <= str_to_ns("16:00:00"), "Invalid first_interval"
        assert self.mkt_close >= str_to_ns("09:30:00") and self.mkt_close <= str_to_ns("16:00:00"), "Invalid mkt_close"
        assert self.timestep_duration > 0 and self.timestep_duration <= str_to_ns("06:30:00"), "Invalid timestep_duration"
        assert isinstance(starting_cash, int) and starting_cash >= 0, "Invalid starting_cash"
        assert isinstance(max_order_size, int) and max_order_size > 0, "Invalid max_order_size"
        assert isinstance(state_history_length, int) and state_history_length >= 0, "Invalid state_history_length"
        assert isinstance(market_data_buffer_length, int) and market_data_buffer_length >= 0, "Invalid market_data_buffer_length"
        assert debug_mode in [True, False], "Invalid debug_mode"
        assert direction in ["BUY", "SELL"], "Invalid direction"
        assert isinstance(parent_order_size, int) and parent_order_size > 0, "Invalid parent_order_size"
        assert self.execution_window > 0 and self.execution_window <= str_to_ns("06:30:00"), "Invalid execution_window"

        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(self.background_config.build_config, background_config_args),
            wakeup_interval_generator=ConstantTimeGenerator(step_duration=self.timestep_duration),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
        )

        self.num_actions = 3
        self.action_space = gym.spaces.Discrete(self.num_actions)
        for i in range(self.num_actions):
            self.custom_metrics_tracker.action_counter[f"action_{i}"] = 0

        num_ns_episode = self.first_interval + self.execution_window
        step_length = self.timestep_duration
        num_max_steps_per_episode = num_ns_episode / step_length
        logger.debug(f"Num max steps per episode: {num_max_steps_per_episode}")
        self.custom_metrics_tracker.num_max_steps_per_episode = num_max_steps_per_episode

        self.num_state_features = 8 + self.state_history_length - 1
        self.state_highs = np.array([2, 2, 4, 1, 1, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max] + 
                                  (self.state_history_length - 1) * [np.finfo(np.float32).max], dtype=np.float32).reshape(self.num_state_features, 1)
        self.state_lows = np.array([-2, -2, -4, 0, 0, np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min] + 
                                 (self.state_history_length - 1) * [np.finfo(np.float32).min], dtype=np.float32).reshape(self.num_state_features, 1)
        self.observation_space = gym.spaces.Box(self.state_lows, self.state_highs, shape=(self.num_state_features, 1), dtype=np.float32)
        self.previous_marked_to_market = self.starting_cash

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action: int) -> List[Dict[str, Any]]:
        """Map Gym action to ABIDES simulator orders.

        Parameters
        ----------
        action : int
            The action index (0: MKT, 1: LMT, 2: Hold).

        Returns
        -------
        List[Dict[str, Any]]
            List of order dictionaries for the simulator.

        Raises
        ------
        ValueError
            If action is not in [0, 1, 2].
        """
        self.custom_metrics_tracker.action_counter[f"action_{action}"] += 1
        if action == 0:
            order_size = self._get_strategy_order_size()
            return [{"type": "CCL_ALL"}, {"type": "MKT", "direction": self.direction, "size": order_size}]
        elif action == 1:
            order_size = self._get_strategy_order_size()
            return [{"type": "CCL_ALL"}, {"type": "LMT", "direction": self.direction, "size": order_size, "limit_price": self.near_touch}]
        elif action == 2:
            return []
        else:
            raise ValueError(f"Action {action} is not supported")

    def _get_strategy_order_size(self):
        """Calculate the order size based on the selected strategy.

        Returns
        -------
        int
            The calculated order size, bounded by max_order_size.

        Notes
        -----
        Uses time_pct, holdings_pct, executed_qty, and volume from the tracker.
        """
        time_pct = self.custom_metrics_tracker.time_pct
        holdings_pct = self.custom_metrics_tracker.holdings_pct
        executed_qty = self.custom_metrics_tracker.executed_quantity
        volume = self.custom_metrics_tracker.volume
        if hasattr(self, 'strategy'):
            strategy = self.strategy
        else:
            strategy = 'TWAP'
        if strategy == "TWAP":
            target_qty = time_pct * self.parent_order_size
            return int(max(1, min(self.max_order_size, abs(target_qty - executed_qty))))
        elif strategy == "VWAP":
            step_idx = min(int(time_pct * (self.execution_window // self.timestep_duration)), 9)
            target_qty = (step_idx / 10) * self.parent_order_size
            return int(max(1, min(self.max_order_size, abs(target_qty - executed_qty))))
        elif strategy == "IS":
            return int(max(1, min(self.max_order_size / 2, self.parent_order_size - abs(executed_qty))))
        elif strategy == "POV":
            target_qty = 0.1 * volume
            return int(max(1, min(self.max_order_size, abs(target_qty - executed_qty))))
        return 1

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Convert raw state to a processed state vector.

        Parameters
        ----------
        raw_state : Dict[str, Any]
            Raw simulation data from the environment.

        Returns
        -------
        ndarray
            Processed state vector of shape (num_state_features, 1).

        Notes
        -----
        Computes features like holdings_pct, time_pct, and market imbalances with validation.
        Handles empty parsed_mkt_data deque with default values and caps negative holdings.
        """
        logger.debug(f"Raw state keys: {raw_state.keys()}")
        bids = raw_state["parsed_mkt_data"]["bids"] if raw_state["parsed_mkt_data"]["bids"] else []
        asks = raw_state["parsed_mkt_data"]["asks"] if raw_state["parsed_mkt_data"]["asks"] else []
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"] if raw_state["parsed_mkt_data"]["last_transaction"] else []
        if not bids or not asks or not last_transactions:
            logger.warning("Empty parsed_mkt_data detected, using default values")
            imbalance_all = 0.0
            imbalance_5 = 0.0
            mid_price = last_transactions[-1] if last_transactions else 1000.0  # Default mid-price
        else:
            imbalances_all = [markets_agent_utils.get_imbalance(b, a, depth=None) for b, a in zip(bids, asks)]
            imbalance_all = imbalances_all[-1] if imbalances_all else 0.0
            imbalances_5 = [markets_agent_utils.get_imbalance(b, a, depth=5) for b, a in zip(bids, asks)]
            imbalance_5 = imbalances_5[-1] if imbalances_5 else 0.0
            mid_prices = [markets_agent_utils.get_mid_price(b, a, lt) for b, a, lt in zip(bids, asks, last_transactions)]
            mid_price = mid_prices[-1] if mid_prices else 0.0
        holdings = raw_state["internal_data"]["holdings"] if isinstance(raw_state["internal_data"]["holdings"], (list, np.ndarray)) else [raw_state["internal_data"]["holdings"]]
        holdings[-1] = max(0, holdings[-1])  # Cap negative holdings to prevent drops
        logger.debug(f"Holdings type: {type(raw_state['internal_data']['holdings'])}, value: {holdings}")
        holdings_pct = holdings[-1] / self.parent_order_size if self.parent_order_size > 0 else 0.0
        mkt_open = raw_state["internal_data"]["mkt_open"][-1] if isinstance(raw_state["internal_data"]["mkt_open"], (list, np.ndarray)) else raw_state["internal_data"]["mkt_open"]
        current_time = raw_state["internal_data"]["current_time"][-1] if isinstance(raw_state["internal_data"]["current_time"], (list, np.ndarray)) else raw_state["internal_data"]["current_time"]
        time_from_parent_arrival = float(current_time) - float(mkt_open) - float(self.first_interval)
        time_pct = time_from_parent_arrival / float(self.execution_window) if self.execution_window > 0 else 0.0
        diff_pct = holdings_pct - time_pct
        if self.step_index == 0 and mid_price > 0:
            self.entry_price = mid_price
        book = raw_state["parsed_mkt_data"]["bids"][-1] if self.direction == "BUY" else raw_state["parsed_mkt_data"]["asks"][-1]
        self.near_touch = book[0][0] if len(book) > 0 else last_transactions[-1] if last_transactions else self.entry_price
        price_impact = np.log(mid_price / self.entry_price) if self.direction == "BUY" and self.entry_price > 0 else np.log(self.entry_price / mid_price) if self.entry_price > 0 else 0.0
        best_bids = [bids[0][0] if len(bids) > 0 else mid_price for bids in [bids[-1]] if bids]
        best_asks = [asks[0][0] if len(asks) > 0 else mid_price for asks in [asks[-1]] if asks]
        spreads = np.array(best_asks) - np.array(best_bids) if best_bids and best_asks else np.array([0.0])
        spread = float(spreads[-1]) if spreads.size > 0 else 0.0
        direction_features = np.array([mid_price - lt for lt in last_transactions[-1:]]) if last_transactions else np.array([0.0])
        direction_feature = float(direction_features[-1]) if direction_features.size > 0 else 0.0
        volume = sum(b[1] for b in bids[-1][:1]) + sum(a[1] for a in asks[-1][:1]) if bids and asks else 0.0
        returns = np.diff([mid_price] + mid_prices[:-1]) if mid_prices else np.array([0.0])
        padded_returns = np.zeros(self.state_history_length - 1)
        padded_returns[-len(returns):] = returns if len(returns) > 0 else padded_returns
        self.custom_metrics_tracker.holdings_pct = float(holdings_pct)
        self.custom_metrics_tracker.time_pct = float(time_pct)
        self.custom_metrics_tracker.diff_pct = float(diff_pct)
        self.custom_metrics_tracker.imbalance_all = float(imbalance_all)
        self.custom_metrics_tracker.imbalance_5 = float(imbalance_5)
        self.custom_metrics_tracker.price_impact = float(price_impact)
        self.custom_metrics_tracker.spread = float(spread)
        self.custom_metrics_tracker.direction_feature = float(direction_feature)
        self.custom_metrics_tracker.volume = float(volume)
        computed_state = np.array([holdings_pct, time_pct, diff_pct, imbalance_all, imbalance_5, price_impact, spread, direction_feature] + 
                                padded_returns.tolist(), dtype=np.float32)
        self.step_index += 1
        logger.debug(f"State computed: holdings_pct={holdings_pct}, time_pct={time_pct}, executed_qty={self.custom_metrics_tracker.executed_quantity}")
        return computed_state.reshape(self.num_state_features, 1)

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """Compute the reward based on trade execution.

        Parameters
        ----------
        raw_state : Dict[str, Any]
            Raw simulation data from the environment.

        Returns
        -------
        float
            The computed reward value.

        Notes
        -----
        Updates trade_schedule and calculates PNL based on executed orders, ensuring monotonic increases.
        """
        entry_price = self.entry_price
        inter_wakeup_executed_orders = raw_state["internal_data"]["inter_wakeup_executed_orders"]
        if len(inter_wakeup_executed_orders) > 0:
            executed_qty = self.custom_metrics_tracker.executed_quantity
            current_time_pct = self.custom_metrics_tracker.time_pct
            new_executed_qty = sum(order.quantity for order in inter_wakeup_executed_orders) if self.direction == "BUY" else -sum(order.quantity for order in inter_wakeup_executed_orders)
            executed_qty = max(executed_qty, executed_qty + new_executed_qty)  # Ensure monotonic increase
            self.custom_metrics_tracker.executed_quantity = executed_qty
            if not any(t[1] == executed_qty for t in self.custom_metrics_tracker.trade_schedule):
                self.custom_metrics_tracker.trade_schedule.append((current_time_pct, executed_qty))
                logger.debug(f"Updated trade_schedule: {self.custom_metrics_tracker.trade_schedule}")
        if len(inter_wakeup_executed_orders) == 0:
            pnl = 0.0
        else:
            pnl = sum((entry_price - order.fill_price) * order.quantity for order in inter_wakeup_executed_orders) if self.direction == "BUY" else sum((order.fill_price - entry_price) * order.quantity for order in inter_wakeup_executed_orders)
        self.pnl = float(pnl)
        self.custom_metrics_tracker.execution_cost += abs(pnl) / self.parent_order_size if self.parent_order_size > 0 else 0.0
        reward = pnl / self.parent_order_size if self.parent_order_size > 0 else 0.0
        self.custom_metrics_tracker.slippage_reward = float(reward)
        logger.debug(f"Reward computed: pnl={pnl}, reward={reward}, executed_qty={self.custom_metrics_tracker.executed_quantity}")
        return reward

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """Compute the final reward update at episode end.

        Parameters
        ----------
        raw_state : Dict[str, Any]
            Raw simulation data from the environment.

        Returns
        -------
        float
            The update reward value.

        Notes
        -----
        Applies penalties or rewards based on execution quantity with validation.
        """
        holdings = raw_state["internal_data"]["holdings"] if isinstance(raw_state["internal_data"]["holdings"], (list, np.ndarray)) else [raw_state["internal_data"]["holdings"]]
        logger.debug(f"Update reward holdings: {holdings}")
        parent_order_size = self.parent_order_size
        if (self.direction == "BUY") and (holdings[-1] >= parent_order_size):
            update_reward = abs(holdings[-1] - parent_order_size) * self.too_much_reward_update
        elif (self.direction == "BUY") and (holdings[-1] < parent_order_size):
            update_reward = abs(holdings[-1] - parent_order_size) * self.not_enough_reward_update
        elif (self.direction == "SELL") and (holdings[-1] <= -parent_order_size):
            update_reward = abs(holdings[-1] - parent_order_size) * self.too_much_reward_update
        elif (self.direction == "SELL") and (holdings[-1] > -parent_order_size):
            update_reward = abs(holdings[-1] - parent_order_size) * self.not_enough_reward_update
        else:
            update_reward = self.just_quantity_reward_update
        update_reward = update_reward / parent_order_size if parent_order_size > 0 else 0.0
        self.custom_metrics_tracker.late_penalty_reward = float(update_reward)
        logger.debug(f"Update reward computed: update_reward={update_reward}, holdings[-1]={holdings[-1]}")
        return update_reward

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """Determine if the episode is done.

        Parameters
        ----------
        raw_state : Dict[str, Any]
            Raw simulation data from the environment.

        Returns
        -------
        bool
            True if episode is done, False otherwise.

        Notes
        -----
        Checks for order completion or time limit with validation.
        """
        holdings = raw_state["internal_data"]["holdings"] if isinstance(raw_state["internal_data"]["holdings"], (list, np.ndarray)) else [raw_state["internal_data"]["holdings"]]
        logger.debug(f"Done holdings: {holdings}")
        parent_order_size = self.parent_order_size
        current_time = raw_state["internal_data"]["current_time"][-1] if isinstance(raw_state["internal_data"]["current_time"], (list, np.ndarray)) else raw_state["internal_data"]["current_time"]
        mkt_open = raw_state["internal_data"]["mkt_open"][-1] if isinstance(raw_state["internal_data"]["mkt_open"], (list, np.ndarray)) else raw_state["internal_data"]["mkt_open"]
        time_limit = float(mkt_open) + float(self.first_interval) + float(self.execution_window)
        done = ((self.direction == "BUY" and holdings[-1] >= parent_order_size) or
                (self.direction == "SELL" and holdings[-1] <= -parent_order_size) or
                (float(current_time) >= time_limit))
        self.custom_metrics_tracker.executed_quantity = max(0, holdings[-1]) if self.direction == "BUY" else max(0, -holdings[-1])  # Prevent negative quantities
        self.custom_metrics_tracker.remaining_quantity = max(0, parent_order_size - self.custom_metrics_tracker.executed_quantity)
        logger.debug(f"Done check: done={done}, executed_qty={self.custom_metrics_tracker.executed_quantity}, remaining_qty={self.custom_metrics_tracker.remaining_quantity}")
        return done

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate info dictionary for debugging or logging.

        Parameters
        ----------
        raw_state : Dict[str, Any]
            Raw simulation data from the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing market and execution metrics.

        Notes
        -----
        Includes trade details and custom metrics if debug_mode is False.
        """
        holdings = raw_state["internal_data"]["holdings"] if isinstance(raw_state["internal_data"]["holdings"], (list, np.ndarray)) else [raw_state["internal_data"]["holdings"]]
        logger.debug(f"Info holdings: {holdings}")
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]
        bids = raw_state["parsed_mkt_data"]["bids"]
        best_bid = bids[0][0] if len(bids) > 0 else last_transaction
        asks = raw_state["parsed_mkt_data"]["asks"]
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction
        current_time = raw_state["internal_data"]["current_time"][-1] if isinstance(raw_state["internal_data"]["current_time"], (list, np.ndarray)) else raw_state["internal_data"]["current_time"]
        info = {
            "last_transaction": float(last_transaction),
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "current_time": float(current_time),
            "holdings": float(holdings[-1]),
            "parent_size": float(self.parent_order_size),
            "pnl": float(self.pnl),
            "reward": float(self.pnl / self.parent_order_size if self.parent_order_size > 0 else 0.0),
            "volume": self.custom_metrics_tracker.volume,
            "execution_cost": self.custom_metrics_tracker.execution_cost,
            "trade_schedule": self.custom_metrics_tracker.trade_schedule,
        }
        if not self.debug_mode:
            info = asdict(self.custom_metrics_tracker)
        return info