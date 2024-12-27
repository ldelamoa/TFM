# TFM Stock Trading Environment

This repository contains the implementation of a custom environment designed for stock trading simulations using Reinforcement Learning (RL). The environment is tailored to train agents on making trading decisions (buy, sell, or hold) based on market data, while optimizing strategies for profit and risk management.

## Features

- **Custom Trading Environment**: Built on OpenAI Gym for flexibility and compatibility with RL frameworks.
- **Market Simulation**: Handles market interactions, including price movements, transaction costs, and portfolio updates.
- **Agent Training**: Facilitates training RL agents for automated trading strategies.
- **Metrics and Visualization**: Tracks performance metrics like rewards, portfolio value, and risk over time.

## Installation

Clone the repository:

```bash
git clone https://github.com/al118345/TFM_stock.git
cd TFM_stock
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Environment

1. Prepare your market data in CSV format (ensure it includes columns like `date`, `open`, `high`, `low`, `close`, and `volume`).
2. Load the environment and initialize it with your data.

```python
from trading_environment import TradingEnvironment

# Example usage:
env = TradingEnvironment(data="path_to_your_data.csv")
```

3. Train your RL agent using libraries such as Stable-Baselines3 or custom implementations.

### Example Workflow

- Load historical stock data.
- Preprocess the data to match the environment's format.
- Define and train an RL agent.
- Evaluate the agent's performance.

## Repository Structure

- `trading_environment.py`: Core implementation of the custom trading environment.
- `utils/`: Helper functions for data preprocessing and performance analysis.
- `examples/`: Example scripts demonstrating how to use the environment.
- `requirements.txt`: List of required Python packages.

## Contributing

Contributions are welcome! If you have ideas for improvements or encounter issues, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Inspired by OpenAI Gym and various reinforcement learning frameworks.
- Thanks to contributors and the community for their valuable feedback and ideas.

---

For further questions or collaborations, please contact the repository owner or open an issue.

