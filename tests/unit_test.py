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
    assert stats.total_transaction_costs == 0  # mock trades use default cost
    print("✓ Statistics calculation test passed")


def test_transaction_costs_reduce_pnl():
    """Test that transaction costs are deducted from round-trip P&L."""
    def close_with_cost(cost_pct):
        trader = TightAggressiveTrader()
        trader.transaction_cost_pct = cost_pct
        position = {
            'entry_price': 100.0, 'entry_period': 1,
            'size': 10.0, 'position_type': 'LONG', 'asset': 'AAPL',
        }
        trader.open_positions.append(position)
        trader.close_position(position, exit_price=110.0, period=5)
        return trader

    free = close_with_cost(0.0)
    costly = close_with_cost(0.01)

    # Raw P&L is (110-100)*10 = 100; cost = 2 * 0.01 * 100 * 10 = 20
    assert free.trades[0].profit_loss == 100.0
    assert costly.trades[0].profit_loss == 80.0
    assert costly.trades[0].transaction_cost == 20.0
    assert costly.get_stats().total_transaction_costs == 20.0
    assert costly.get_stats().total_pnl < free.get_stats().total_pnl
    print("✓ Transaction costs test passed")


def test_position_size_converted_to_shares():
    """execute_trade must convert dollar risk to shares at entry price."""
    trader = TightAggressiveTrader()  # risk 2% + 0.8*3% + 0.2*2% = 4.8% -> $480
    trader.execute_trade(100.0, MarketCondition.BULLISH, period=1)
    pos = trader.open_positions[0]
    assert abs(pos['size'] - 4.80) < 1e-9  # $480 / $100 per share
    trader.close_position(pos, exit_price=110.0, period=2)
    trade = trader.trades[0]
    assert abs(trade.transaction_cost - 0.0) < 1e-9  # cost_pct still 0
    assert abs(trade.profit_loss - 10.0 * 4.80) < 1e-9  # $48 raw P&L
    print("✓ Share conversion test passed")


def test_equity_curve_is_daily():
    """Equity curve must be one point per market period, not per trade."""
    np.random.seed(42)
    import random as _random
    _random.seed(42)
    simulator = MarketSimulator(periods=50, initial_price=100)
    simulator.generate_market()
    traders = [
        LoosePassiveTrader(), LooseAggressiveTrader(),
        TightPassiveTrader(), TightAggressiveTrader(),
    ]
    simulator.run_simulation(traders)

    for trader in traders:
        assert len(trader.equity_curve) == 51  # initial + 50 periods
        equity = np.array(trader.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        assert drawdown.min() >= -1.0  # equity never goes below zero
    print("✓ Daily equity curve test passed")


if __name__ == "__main__":
    print("Running tests...\n")
    test_market_generation()
    test_trader_initialization()
    test_simulation_runs()
    test_transaction_costs_reduce_pnl()
    test_position_size_converted_to_shares()
    test_equity_curve_is_daily()
    test_statistics_calculation()
    print("\n✓ All tests passed!")