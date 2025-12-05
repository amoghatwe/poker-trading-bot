# Contributing to Poker Trading Styles Simulator

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Python version and OS information
- Code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- Clear description of the proposed feature
- Rationale for why it would be useful
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add or update tests if applicable
5. Update documentation as needed
6. Ensure all tests pass
7. Commit your changes with clear, descriptive messages
8. Push to your fork
9. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/poker-trading-simulator.git
cd poker-trading-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public methods
- Keep functions focused and modular
- Add comments for complex logic

## Testing

Before submitting a PR:
```bash
# Run the main simulation
python poker_trading.py

# Ensure no errors occur
```

## Areas for Contribution

- **New Trading Styles**: Implement additional behavioral archetypes
- **Technical Indicators**: Add moving averages, RSI, MACD, etc.
- **Visualization**: Create charts for equity curves and performance
- **Optimization**: Improve simulation speed
- **Testing**: Add unit tests and integration tests
- **Documentation**: Improve examples and tutorials
- **Real Data Integration**: Connect to market data APIs

## Questions?

Feel free to open an issue for any questions about contributing!
