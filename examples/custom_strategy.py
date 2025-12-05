"""
Example: Custom Trading Strategy
=================================

This example demonstrates how to create a custom trading strategy
by extending the TradingStyle base class.
"""

from poker_trading import *
import numpy as np


class MomentumTrader(TradingStyle):
    """
    A custom momentum-based trading style.

    Enters trades based on recent price momentum and volatility.
    """

    def __init__(self):
        super().__init__(
            name="Momentum Trader",
            aggressiveness=0.6,
            responsiveness=0.4,
            initial_capital=10000
        )
        self.lookback_period = 10

    def should_enter_trade(self, market_condition, signal_strength):
        """
        Custom entry logic based on momentum and volatility.
        """
        # Base entry logic
        base_decision = super().should_enter_trade(market_condition, signal_strength)

        # Add momentum filter - only trade in strong trends
        if signal_strength < 0.6:
            return False

        return base_decision

    def position_size(self):
        """
        Dynamic position sizing based on current capital.
        """
        # Use Kelly Criterion-inspired sizing
        base_size = super().position_size()

        # Scale down if capital has decreased
        capital_ratio = self.capital / self.initial_capital
        adjusted_size = base_size * min(capital_ratio, 1.0)

        return adjusted_size


def main():
    """Run custom strategy comparison."""
    print("=" * 80)
    print("CUSTOM MOMENTUM STRATEGY EXAMPLE")
    print("=" * 80)

    # Create market
    np.random.seed(123)
    simulator = MarketSimulator(periods=252, initial_price=100)
    simulator.generate_market()

    # Compare custom strategy against baseline
    traders = [
        TightAggressiveTrader(),  # Baseline
        MomentumTrader()          # Custom strategy
    ]

    # Run simulation
    simulator.run_simulation(traders)

    # Display results
    print_comparison_table(traders)
    print_individual_stats(traders)


if __name__ == "__main__":
    main()
