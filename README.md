# Poker Trading Styles Simulator 🃏📈

A Python-based trading simulator that applies poker playing styles to quantitative trading strategies. This project explores how different poker playing  archetypes perform in simulated financial markets.

## Project Overview

This simulator models four distinct trading styles based on a 2x2 matrix:

| Style | Aggressiveness | Responsiveness | Poker Analogy | Trading Behavior |
|-------|---------------|----------------|---------------|------------------|
| **Loose-Passive** | Low (20%) | High (80%) | Calling station | Responds to many weak signals but trades infrequently |
| **Loose-Aggressive** | High (80%) | High (80%) | LAG player | High frequency trading on weak signals |
| **Tight-Passive** | Low (20%) | Low (20%) | Nit/Rock | Selective entries, conservative execution |
| **Tight-Aggressive** | High (80%) | Low (20%) | TAG player | Selective entries, decisive execution |

### Key Dimensions

- **Aggressiveness (Passive ↔ Aggressive)**: How *frequently* trades are executed
- **Responsiveness (Tight ↔ Loose)**: *Signal sensitivity* and entry criteria

## Features

- **Four Trading Archetypes**: Each with distinct entry/exit logic and position sizing
- **Market Simulation**: Geometric Brownian motion for realistic price movements
- **Comprehensive Metrics**: Win rate, P&L, Sharpe ratio, max drawdown, profit factor
- **Comparative Analysis**: Side-by-side performance comparison
- **Reproducible Results**: Seeded random number generation


## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/amoghatwe/poker-trading-simulator.git
cd poker-trading-simulator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from poker_trading import *

# Create market simulator
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
print_comparison_table(traders)
```

### Running the Default Simulation

```bash
python poker_trading.py
```

### Custom Parameters

```python
# Custom trader with specific parameters
custom_trader = TradingStyle(
    name="Balanced Trader",
    aggressiveness=0.5,
    responsiveness=0.5,
    initial_capital=50000
)

# Longer simulation period
simulator = MarketSimulator(periods=1000, initial_price=150)
```

## 📊 Sample Output

Actual output from `python poker_trading.py` (seed 42, 252 days, 10bps/side transaction costs):

```
================================================================================
POKER TRADING STYLES SIMULATOR
================================================================================
Simulating 4 trading styles across 252 trading days

================================================================================
TRADER COMPARISON ANALYSIS
================================================================================
Trader                      Trades    Win %    Total P&L    Return    Max DD Profit Factor      Costs
--------------------------------------------------------------------------------------------
Loose-Passive Trader           158   47.5% $       -129    -1.3%    -3.2%        0.90x $      132
Loose-Aggressive Trader         74   47.3% $        -72    -0.7%    -2.3%        0.91x $       88
Tight-Passive Trader           143   46.2% $       -224    -2.2%    -2.4%        0.77x $       85
Tight-Aggressive Trader         61   47.5% $         47     0.5%    -1.0%        1.10x $       59

================================================================================
PERFORMANCE HIGHLIGHTS
================================================================================

🏆 Best P&L: Tight-Aggressive Trader
   Total Profit: $46.64 (0.5%)

📊 Best Win Rate: Tight-Aggressive Trader
   Win Rate: 47.5% (29/61)

⚡ Best Risk-Adjusted Return: Tight-Aggressive Trader
   Sharpe Ratio: 0.22

✅ VERDICT: Tight-Aggressive Trader still wins after 10bps/side transaction costs
   Most cost-impacted: Loose-Passive Trader ($132 across 158 trades)
```

## 🧠 Methodology

### Trading Logic

Each trader makes decisions based on:

1. **Entry Logic**: 
   - Signal threshold (tight traders require stronger signals)
   - Frequency filter (passive traders enter less often)

2. **Position Sizing**:
   - Base risk: 2% of capital
   - Adjusted by aggressiveness (+0-3%) and looseness (+0-2%)
   - Dollar risk converted to shares at entry price

3. **Exit Logic**:
   - Profit targets (3-8% depending on style)
   - Stop losses (-5% to -10%)
   - Time-based exits (2-8 periods)

### Market Simulation

- **Model**: Geometric Brownian motion
- **Parameters**: 0.05% daily drift, 2% volatility
- **Conditions**: BULLISH, BEARISH, NEUTRAL based on 5-period lookback
- **Equity curve**: marked to market daily (capital + unrealized P&L); Sharpe and drawdown are computed from daily returns

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Total Trades** | Number of completed trades |
| **Win Rate** | Percentage of profitable trades |
| **Total P&L** | Net profit/loss in dollars |
| **Return** | P&L as percentage of initial capital |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |

## 🎓 Insights & Findings

Based on the seeded run (seed 42, 10bps/side) and its zero-cost baseline,
with dollar-accurate P&L (share sizing fixed) and daily mark-to-market equity:

1. **TAG wins on every metric after costs**: best P&L (+$46.64, the only
   positive), best win rate (47.5%), best Sharpe (0.22), smallest drawdown
   (−1.0%). It also led at zero cost (+$105.61) — costs didn't flip the
   ranking, they halved TAG's edge.

2. **Costs flip the loose styles from profit to loss**: LP +$2.78 → −$129.14,
   LAG +$16.55 → −$71.97. Their zero-cost profits were smaller than their
   cost drag ($132 and $88 respectively). This is the classic overtrading
   failure mode — marginal edges don't survive realistic friction.

3. **Total costs track trade count, not style**: LP pays the most ($132 over
   158 trades) despite the smallest average position; TAG pays the least
   ($59 over 61 trades) because it trades least often. Per-trade cost runs
   $0.59–$1.19, ≈20bps on average position notional ($300–$600).

4. **Net P&L impact is path-dependent**: `position_size()` scales with
   capital, so cost drag shrinks subsequent positions. Every style's costed
   P&L is lower than its zero-cost P&L, but by less than its total costs
   (TAG: −$58.97 net vs $58.54 costs) — the sizing feedback cushions part
   of the drag. The cost mechanic itself is exact (unit-tested: $100 → $80
   on $1,000 notional at 1%/side).

5. **Earlier "TAG wins big" sample outputs were artifacts of a sizing bug**:
   the prior code multiplied a dollar position by price change as if it
   were shares, inflating P&L ~100× (returns like −115%). With true dollar
   P&L, all returns here fall in −2.2%..+0.5% and drawdowns in −1%..−3.2%,
   consistent with ~2–5% of capital at risk per trade.

## 🛠️ Extending the Project

### Adding New Trading Styles

```python
class CustomTrader(TradingStyle):
    def __init__(self):
        super().__init__(
            name="Custom Trader",
            aggressiveness=0.6,
            responsiveness=0.3,
            initial_capital=10000
        )

    # Override methods for custom behavior
    def should_enter_trade(self, market_condition, signal_strength):
        # Custom entry logic
        pass
```

### Adding Technical Indicators

```python
def calculate_moving_average(prices, window=20):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

# Integrate into MarketSimulator
```

## 📚 Background

This project was inspired by:
- **Poker Theory**: GTO (Game Theory Optimal) and exploitative play
- **Behavioral Finance**: How biases affect trading performance
- **Quantitative Trading**: Systematic strategy development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Special word of thanks to Prof. (Dr.) Ankush Garg who helped me flesh out the initial game theory aspect of the project, without his help I would not have been able to start the project!
- Inspired by poker strategy literature (Harrington, Sklansky, etc.)
- Quantitative finance community for trading metrics and best practices
- Behavioral economics research on decision-making under uncertainty

## Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue on GitHub
- Email: [amoghatwe@gmail.com]

## 🔮 Future Enhancements

- [ ] Add machine learning-based adaptive strategies
- [ ] Implement multi-asset portfolio simulation
- [x] Include transaction costs and slippage (round-trip cost model, configurable via `transaction_cost_pct`)

---

**Note**: This is a simulation for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough research before implementing real trading strategies.
