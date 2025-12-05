"""
Basic tests for poker_trading module.
"""

import sys
sys.path.append('..')

from poker_trading import *
import numpy as np


def test_market_generation():
    """Test that market simulator generates prices correctly."""
    simulator = MarketSimulator(periods=100, initial_price=100)
    prices = simulator.generate_market()

    assert len(prices) == 101  # initial + 100 periods
    assert prices[0] == 100
    assert all(p > 0 for p in prices)
    print("✓ Market generation test passed")


def test_trader_initialization():
    """Test that all trader types initialize correctly."""
    traders = [
        LoosePassiveTrader(),
        LooseAggressiveTrader(),
        TightPassiveTrader(),
        TightAggressiveTrader()
    ]

    for trader in traders:
        assert trader.capital == 10000
        assert len(trader.trades) == 0
        assert len(trader.open_positions) == 0
    print("✓ Trader initialization test passed")


def test_simulation_runs():
    """Test that simulation completes without errors."""
    np.random.seed(42)

    simulator = MarketSimulator(periods=50, initial_price=100)
    simulator.generate_market()

    traders = [
        TightAggressiveTrader(),
        LooseAggressiveTrader()
    ]

    simulator.run_simulation(traders)

    # Check that some trades were made
    total_trades = sum(len(t.trades) for t in traders)
    assert total_trades > 0
    print(f"✓ Simulation test passed ({total_trades} total trades)")


def test_statistics_calculation():
    """Test that statistics are calculated correctly."""
    trader = TightAggressiveTrader()

    # Add mock trades
    trader.trades = [
        Trade(1, "AAPL", "LONG", 100, 105, 100, 500, 5),
        Trade(2, "AAPL", "SHORT", 105, 103, 100, 200, 3),
        Trade(3, "AAPL", "LONG", 103, 100, 100, -300, 4),
    ]

    stats = trader.get_stats()

    assert stats.total_trades == 3
    assert stats.winning_trades == 2
    assert stats.losing_trades == 1
    assert stats.total_pnl == 400
    print("✓ Statistics calculation test passed")


if __name__ == "__main__":
    print("Running tests...\n")
    test_market_generation()
    test_trader_initialization()
    test_simulation_runs()
    test_statistics_calculation()
    print("\n✓ All tests passed!")