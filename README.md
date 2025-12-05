<<<<<<< HEAD
# Poker Agents Trading Bot

This project is a way to evaluate the best strategies for trading on financial markets. It will revolve around four main playstyles: Loose-Aggressive, Loose-Passive, Tight-Aggressive, and Tight-Passive.

The project will start off evaluating these strategies on a market-simulation.

The project is currently in it's most nascent stage, I will be evaluating the mathematical literature behind these strategies and then apply them through Python and/or C++ at a later stage.
=======
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

```
================================================================================
POKER TRADING STYLES SIMULATOR
================================================================================
Simulating 4 trading styles across 252 trading days

================================================================================
TRADER COMPARISON ANALYSIS
================================================================================
Trader                       Trades    Win %    Total P&L    Return    Max DD  Profit Factor
------------------------------------------------------------------------------------
Loose-Passive Trader             45    51.1%      $1,234    12.3%    -8.2%         1.45x
Loose-Aggressive Trader         189    48.7%      $-876    -8.8%   -15.3%         0.82x
Tight-Passive Trader             12    66.7%      $2,145    21.5%    -4.1%         3.21x
Tight-Aggressive Trader          38    63.2%      $3,567    35.7%    -5.8%         2.87x

================================================================================
PERFORMANCE HIGHLIGHTS
================================================================================

🏆 Best P&L: Tight-Aggressive Trader
   Total Profit: $3,567.00 (35.7%)

📊 Best Win Rate: Tight-Passive Trader
   Win Rate: 66.7% (8/12)

⚡ Best Risk-Adjusted Return: Tight-Aggressive Trader
   Sharpe Ratio: 1.89
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

3. **Exit Logic**:
   - Profit targets (3-8% depending on style)
   - Stop losses (-5% to -10%)
   - Time-based exits (2-8 periods)

### Market Simulation

- **Model**: Geometric Brownian motion
- **Parameters**: 0.05% daily drift, 2% volatility
- **Conditions**: BULLISH, BEARISH, NEUTRAL based on 5-period lookback

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Total Trades** | Number of completed trades |
| **Win Rate** | Percentage of profitable trades |
| **Total P&L** | Net profit/loss in dollars |
| **Return** | P&L as percentage of initial capital |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Profit Factor** | Gross profit / Gross loss |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |

## 🎓 Insights & Findings

Based on typical simulation runs:

1. **Tight-Aggressive** (TAG) style typically outperforms:
   - Highest absolute returns
   - Best risk-adjusted performance
   - Moderate trade frequency

2. **Loose-Aggressive** often underperforms:
   - High transaction costs from overtrading
   - Low signal quality leads to poor outcomes

3. **Tight-Passive** shows promise but limits upside:
   - High win rate but fewer opportunities
   - Low drawdowns but capped returns

4. **Loose-Passive** exhibits inconsistent results:
   - Moderate performance
   - Lacks clear edge

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
- Email: [mailto:amoghatwe@gmail.com](mailto:amoghatwe@gmail.com)

## 🔮 Future Enhancements

- [ ] Add machine learning-based adaptive strategies
- [ ] Implement multi-asset portfolio simulation
- [ ] Include transaction costs and slippage
- [ ] Add visualization dashboard (matplotlib/plotly)
- [ ] Implement Monte Carlo simulations for robustness testing
- [ ] Add regime detection (bull/bear market identification)
- [ ] Integrate real market data (via yfinance or similar)
- [ ] Create web interface for interactive simulations

---

**Note**: This is a simulation for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough research before implementing real trading strategies.
>>>>>>> a828da8 (Finished Code, Poker Trading Bot)
