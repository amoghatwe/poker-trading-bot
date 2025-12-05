"""
Poker Trading Styles Simulator
===============================

A Python simulator that emulates four poker-inspired trading styles in quantitative trading.
This project explores how different behavioral archetypes perform in simulated market conditions.

Trading Styles (2x2 Matrix):
- Loose/Tight: Market Responsiveness (signal sensitivity)
- Passive/Aggressive: Trading Frequency (execution rate)

The Four Archetypes:
1. LOOSE-PASSIVE: Responds to many signals but trades infrequently
2. LOOSE-AGGRESSIVE: Responds to many signals and trades frequently
3. TIGHT-PASSIVE: Selective about signals and trades infrequently
4. TIGHT-AGGRESSIVE: Selective about signals but acts decisively (optimal style)

Author: Amogh Satyajit Atwe
Date: December 2025
License: MIT
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class MarketCondition(Enum):
    """Represents market direction."""

    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class Trade:
    """Represents a single executed trade."""

    timestamp: int
    asset: str
    position_type: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    size: float
    profit_loss: float
    duration: int  # periods held

    def __str__(self) -> str:
        return (f"{self.position_type} @ {self.entry_price:.2f} → "
                f"{self.exit_price:.2f} | P&L: ${self.profit_loss:.2f}")

    def roi(self) -> float:
        """Return on investment for this trade."""
        return self.profit_loss / (self.entry_price * self.size)


@dataclass
class TraderStats:
    """Complete trading statistics."""

    trader_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float

    def __str__(self) -> str:
        return f"""
{self.trader_name} Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:   {self.total_trades}
Win Rate:       {self.win_rate:.1%}
Total P&L:      ${self.total_pnl:,.2f}
Max Drawdown:   {self.max_drawdown:.1%}
Profit Factor:  {self.profit_factor:.2f}x
Avg Win:        ${self.avg_win:.2f}
Avg Loss:       ${self.avg_loss:.2f}
Sharpe Ratio:   {self.sharpe_ratio:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ============================================================================
# BASE TRADING STYLE CLASS
# ============================================================================

class TradingStyle(ABC):
    """
    Base class for all trading styles.

    Key Parameters:
        aggressiveness (0.0-1.0): How often trades are executed
        responsiveness (0.0-1.0): How loose/tight entry criteria are
    """

    def __init__(
        self,
        name: str,
        aggressiveness: float,
        responsiveness: float,
        initial_capital: float = 10000
    ):
        """
        Initialize a trading style.

        Args:
            name: Display name
            aggressiveness: 0.0 (Passive) to 1.0 (Aggressive)
            responsiveness: 0.0 (Tight) to 1.0 (Loose)
            initial_capital: Starting capital
        """
        self.name = name
        self.aggressiveness = aggressiveness
        self.responsiveness = responsiveness
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.open_positions: List[dict] = []

    def should_enter_trade(
        self,
        market_condition: MarketCondition,
        signal_strength: float
    ) -> bool:
        """
        Decision logic for entering a trade.

        - Tight traders need stronger signals (high threshold)
        - Loose traders accept weaker signals (low threshold)
        - Aggressive traders enter more frequently
        - Passive traders are selective about entries

        Args:
            market_condition: Current market direction
            signal_strength: Signal strength from 0.0 to 1.0

        Returns:
            True if trade should be entered, False otherwise
        """
        # Signal threshold: Tight traders (0.2 responsiveness) need 0.9 strength
        # Loose traders (0.8 responsiveness) need 0.5 strength
        signal_threshold = 1.0 - (self.responsiveness * 0.5)

        # Frequency threshold: Aggressive traders have ~70% more entry opportunities
        # Passive traders are much more selective
        frequency_threshold = 1.0 - (self.aggressiveness * 0.7)

        meets_signal = signal_strength >= signal_threshold
        meets_frequency = random.random() < frequency_threshold

        return meets_signal and meets_frequency

    def position_size(self) -> float:
        """
        Calculate position size based on trading style.

        - Aggressive traders: Larger positions, more risk
        - Loose traders: Less careful with position sizing
        - Passive traders: Smaller, more conservative positions
        - Tight traders: More disciplined position sizing

        Returns:
            Position size in dollars
        """
        base_risk = 0.02  # 2% base risk per trade
        aggressive_component = self.aggressiveness * 0.03
        loose_component = self.responsiveness * 0.02
        risk_per_trade = base_risk + aggressive_component + loose_component
        return self.capital * risk_per_trade

    def execute_trade(
        self,
        entry_price: float,
        market_condition: MarketCondition,
        period: int,
        asset: str = "Stock"
    ) -> None:
        """Execute a new trade."""
        size = self.position_size()
        position_type = "LONG" if market_condition == MarketCondition.BULLISH else "SHORT"

        self.open_positions.append({
            'entry_price': entry_price,
            'entry_period': period,
            'size': size,
            'position_type': position_type,
            'asset': asset
        })

    def should_exit_trade(
        self,
        current_price: float,
        unrealized_pnl_pct: float,
        periods_held: int
    ) -> bool:
        """
        Decision logic for exiting a trade.

        - Aggressive traders: Hold longer for bigger moves
        - Passive traders: Exit quicker to lock in small gains
        - Loose traders: More emotional exits (take profits quickly, let losses run)
        - Tight traders: Disciplined stop losses and profit targets

        Args:
            current_price: Current market price
            unrealized_pnl_pct: Unrealized P&L as percentage
            periods_held: Number of periods position has been held

        Returns:
            True if position should be closed, False otherwise
        """
        # Time-based holding period
        # Aggressive: 5-8 periods, Passive: 2-5 periods
        time_threshold = 5 - (self.aggressiveness * 3)

        # Profit taking threshold
        # Loose traders take profits quickly (3-8%)
        # Tight traders hold for bigger moves (3-5%)
        profit_target = 0.03 + (self.responsiveness * 0.05)

        # Stop loss threshold
        # Tight traders cut losses quickly (-5 to -10%)
        # Loose traders let losses run longer
        stop_loss = -0.05 - (self.responsiveness * 0.05)

        # Check exit conditions
        if unrealized_pnl_pct >= profit_target:
            return True  # Hit profit target
        if unrealized_pnl_pct <= stop_loss:
            return True  # Hit stop loss
        if periods_held >= time_threshold:
            # Time-based exit with some randomness
            return random.random() < 0.3

        return False

    def close_position(self, position: dict, exit_price: float, period: int) -> None:
        """Close a position and record the trade."""
        entry_price = position['entry_price']
        size = position['size']
        position_type = position['position_type']

        # Calculate P&L
        if position_type == "LONG":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        # Record trade
        trade = Trade(
            timestamp=period,
            asset=position['asset'],
            position_type=position_type,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            profit_loss=pnl,
            duration=period - position['entry_period']
        )

        self.trades.append(trade)
        self.capital += pnl
        self.equity_curve.append(self.capital)
        self.open_positions.remove(position)

    def get_stats(self) -> TraderStats:
        """Calculate and return comprehensive trading statistics."""
        if not self.trades:
            return TraderStats(
                trader_name=self.name,
                total_trades=0, winning_trades=0, losing_trades=0,
                total_pnl=0, max_drawdown=0, win_rate=0,
                avg_win=0, avg_loss=0, profit_factor=0, sharpe_ratio=0
            )

        pnls = [trade.profit_loss for trade in self.trades]
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)
        total_wins = sum(pnl for pnl in pnls if pnl > 0)
        total_losses = abs(sum(pnl for pnl in pnls if pnl < 0))

        avg_win = total_wins / winning_trades if winning_trades > 0 else 0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Max drawdown calculation
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Sharpe ratio (simplified, assumes risk-free rate = 0)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

        return TraderStats(
            trader_name=self.name,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=sum(pnls),
            max_drawdown=max_drawdown,
            win_rate=winning_trades / len(self.trades) if self.trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe
        )


# ============================================================================
# FOUR TRADING STYLE IMPLEMENTATIONS
# ============================================================================

class LoosePassiveTrader(TradingStyle):
    """
    Loose-Passive Trader
    ────────────────────
    Aggressiveness: LOW (20%) - Trades infrequently
    Responsiveness: HIGH (80%) - Responds to many weak signals

    Poker Analogy: Calls many hands but plays few hands overall.
    Trading Behavior: Enters on weak signals but rarely executes. When in,
                     takes small profits and cuts losses quickly (emotional).
    """

    def __init__(self):
        super().__init__(
            name="Loose-Passive Trader",
            aggressiveness=0.2,
            responsiveness=0.8,
            initial_capital=10000
        )


class LooseAggressiveTrader(TradingStyle):
    """
    Loose-Aggressive Trader
    ───────────────────────
    Aggressiveness: HIGH (80%) - Trades frequently
    Responsiveness: HIGH (80%) - Responds to many weak signals

    Poker Analogy: Plays many hands aggressively (LAG).
    Trading Behavior: Constantly entering on weak signals, taking small
                     profits and losses. High churn, low precision.
    """

    def __init__(self):
        super().__init__(
            name="Loose-Aggressive Trader",
            aggressiveness=0.8,
            responsiveness=0.8,
            initial_capital=10000
        )


class TightPassiveTrader(TradingStyle):
    """
    Tight-Passive Trader
    ───────────────────
    Aggressiveness: LOW (20%) - Trades infrequently
    Responsiveness: LOW (20%) - Tight entry criteria

    Poker Analogy: Only plays premium hands but doesn't bet aggressively.
    Trading Behavior: Selective entries on strong signals, but once in,
                     holds positions too long. Conservative position sizing.
    """

    def __init__(self):
        super().__init__(
            name="Tight-Passive Trader",
            aggressiveness=0.2,
            responsiveness=0.2,
            initial_capital=10000
        )


class TightAggressiveTrader(TradingStyle):
    """
    Tight-Aggressive Trader
    ──────────────────────
    Aggressiveness: HIGH (80%) - Acts decisively when conditions align
    Responsiveness: LOW (20%) - Only enters on strong signals

    Poker Analogy: Plays premium hands aggressively (TAG). The optimal style.
    Trading Behavior: Selective entries on strong signals, larger positions,
                     decisive exits. Best risk-reward profile.
    """

    def __init__(self):
        super().__init__(
            name="Tight-Aggressive Trader",
            aggressiveness=0.8,
            responsiveness=0.2,
            initial_capital=10000
        )


# ============================================================================
# MARKET SIMULATOR
# ============================================================================

class MarketSimulator:
    """Simulates market price movements and runs traders against them."""

    def __init__(self, periods: int = 252, initial_price: float = 100):
        """
        Args:
            periods: Number of trading periods (typically 252 = 1 year)
            initial_price: Starting price
        """
        self.periods = periods
        self.initial_price = initial_price
        self.prices: List[float] = [initial_price]
        self.market_conditions: List[MarketCondition] = []

    def generate_market(self) -> List[float]:
        """
        Generate realistic market data using geometric random walk.
        Price_t = Price_{t-1} * (1 + drift + noise)
        """
        price = self.initial_price

        for i in range(self.periods):
            # Geometric Brownian motion: small random changes
            drift = random.gauss(0.0005, 0.02)  # 0.05% daily drift ± 2% vol
            price *= (1 + drift)
            self.prices.append(price)

            # Determine market condition based on recent trend
            lookback = min(5, len(self.prices) - 1)
            recent_change = (self.prices[-1] - self.prices[-lookback-1]) / self.prices[-lookback-1]

            if recent_change > 0.01:
                condition = MarketCondition.BULLISH
            elif recent_change < -0.01:
                condition = MarketCondition.BEARISH
            else:
                condition = MarketCondition.NEUTRAL

            self.market_conditions.append(condition)

        return self.prices

    def run_simulation(self, traders: List[TradingStyle]) -> None:
        """Execute simulation for all traders against market."""
        for period in range(1, len(self.prices)):
            current_price = self.prices[period]
            market_condition = self.market_conditions[period - 1]

            # Generate signal strength based on price movement
            price_change = (current_price - self.prices[period - 1]) / self.prices[period - 1]
            signal_strength = min(1.0, abs(price_change) * 100)

            for trader in traders:
                # Process exits first
                for position in trader.open_positions[:]:
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    unrealized_pnl_pct = unrealized_pnl / (position['entry_price'] * position['size'])

                    if trader.should_exit_trade(
                        current_price,
                        unrealized_pnl_pct,
                        period - position['entry_period']
                    ):
                        trader.close_position(position, current_price, period)

                # Process entries
                if (trader.capital > 0 and
                    trader.should_enter_trade(market_condition, signal_strength)):
                    trader.execute_trade(current_price, market_condition, period)


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def print_comparison_table(traders: List[TradingStyle]) -> None:
    """Print comprehensive comparison of all traders."""
    stats_list = [trader.get_stats() for trader in traders]

    print("\n" + "="*80)
    print("TRADER COMPARISON ANALYSIS")
    print("="*80)

    headers = ["Trader", "Trades", "Win %", "Total P&L", "Return", "Max DD", "Profit Factor"]
    print(f"{headers[0]:<25} {headers[1]:>8} {headers[2]:>8} {headers[3]:>12} "
          f"{headers[4]:>9} {headers[5]:>9} {headers[6]:>12}")
    print("-" * 80)

    for trader, stats in zip(traders, stats_list):
        pf = f"{stats.profit_factor:.2f}x" if stats.profit_factor != float('inf') else "∞"
        print(f"{stats.trader_name:<25} {stats.total_trades:>8} {stats.win_rate:>7.1%} "
              f"${stats.total_pnl:>11,.0f} {stats.total_pnl/10000:>8.1%} "
              f"{stats.max_drawdown:>8.1%} {pf:>12}")


def print_individual_stats(traders: List[TradingStyle]) -> None:
    """Print detailed statistics for each trader."""
    for trader in traders:
        stats = trader.get_stats()
        print(stats)


def find_best_trader(
    traders: List[TradingStyle],
    metric: str = "pnl"
) -> Tuple[TradingStyle, TraderStats]:
    """
    Find best trader by metric.

    Args:
        traders: List of trading styles
        metric: One of 'pnl', 'win_rate', 'sharpe', 'trades'

    Returns:
        Tuple of (best trader, their stats)
    """
    stats_list = [trader.get_stats() for trader in traders]

    if metric == "pnl":
        best_idx = max(range(len(stats_list)), key=lambda i: stats_list[i].total_pnl)
    elif metric == "win_rate":
        best_idx = max(range(len(stats_list)), key=lambda i: stats_list[i].win_rate)
    elif metric == "sharpe":
        best_idx = max(range(len(stats_list)), key=lambda i: stats_list[i].sharpe_ratio)
    elif metric == "trades":
        best_idx = max(range(len(traders)), key=lambda i: len(traders[i].trades))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return traders[best_idx], stats_list[best_idx]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    print("="*80)
    print("POKER TRADING STYLES SIMULATOR")
    print("="*80)
    print("Simulating 4 trading styles across 252 trading days\n")

    # Create market
    simulator = MarketSimulator(periods=252, initial_price=100)
    simulator.generate_market()

    # Create traders
    traders = [
        LoosePassiveTrader(),
        LooseAggressiveTrader(),
        TightPassiveTrader(),
        TightAggressiveTrader()
    ]

    # Run simulation
    simulator.run_simulation(traders)

    # Display results
    print_individual_stats(traders)
    print_comparison_table(traders)

    # Find winners
    print("\n" + "="*80)
    print("PERFORMANCE HIGHLIGHTS")
    print("="*80)

    best_pnl, stats = find_best_trader(traders, "pnl")
    print(f"\n🏆 Best P&L: {best_pnl.name}")
    print(f"   Total Profit: ${stats.total_pnl:,.2f} ({stats.total_pnl/10000:.1%})")

    best_wr, stats = find_best_trader(traders, "win_rate")
    print(f"\n📊 Best Win Rate: {best_wr.name}")
    print(f"   Win Rate: {stats.win_rate:.1%} ({stats.winning_trades}/{stats.total_trades})")

    best_sharpe, stats = find_best_trader(traders, "sharpe")
    print(f"\n⚡ Best Risk-Adjusted Return: {best_sharpe.name}")
    print(f"   Sharpe Ratio: {stats.sharpe_ratio:.2f}")

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
