** Note: This README was created with the assistance of AI to provide a comprehensive and professional documentation for the repository. The AI helped identify missing components in the original repository and created additional files and documentation to ensure a complete and functional  workflow.ALL the code has been done by the repository owner that is https://github.com/Rishirajbal with the assistance of deepseeck .

# MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading

This repository contains the implementation of the KDD 2024 paper ["MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading"](https://arxiv.org/abs/2406.14537). The project leverages advanced reinforcement learning techniques to optimize trading strategies in high-frequency trading environments.

## Project Overview

MacroHFT is a novel approach to high-frequency trading that combines memory augmentation with context-aware reinforcement learning. The model is designed to make trading decisions by analyzing market data and technical indicators while maintaining memory of past market states to improve decision-making in volatile market conditions.

The implementation uses Deep Q-Networks (DQN) with several enhancements:
- Prioritized Experience Replay for efficient learning
- Double DQN architecture to reduce overestimation bias
- Layer normalization and dropout for improved generalization
- Adaptive learning rate scheduling

## Technical Architecture

### Core Components

1. **DQN Neural Network**: A multi-layer neural network with layer normalization and dropout for robust feature extraction and decision-making.

2. **Prioritized Replay Buffer**: An advanced memory mechanism that prioritizes important experiences for more efficient learning, using importance sampling to correct for bias.

3. **Trading Environment**: A custom OpenAI Gym-like environment that simulates trading with realistic constraints including transaction costs.

4. **Technical Indicator Engine**: Automatically calculates and normalizes technical indicators like RSI, MACD, and ATR to provide the model with relevant market features.

### Neural Network Architecture

```
DQN(
  (net): Sequential(
    (0): Linear(input_dim, 256)
    (1): LayerNorm(256)
    (2): LeakyReLU()
    (3): Dropout(0.2)
    (4): Linear(256, 128)
    (5): LayerNorm(128)
    (6): LeakyReLU()
    (7): Dropout(0.2)
    (8): Linear(128, output_dim)
  )
)
```

## Key Features

### Memory Augmentation

The model implements a sophisticated memory mechanism through its prioritized experience replay buffer. This allows the agent to:
- Store and recall important trading experiences
- Prioritize learning from significant market events
- Maintain a balance between exploration and exploitation
- Adapt to changing market conditions

### Context Awareness

The trading agent maintains awareness of:
- Current position (long or cash)
- Historical price movements
- Technical indicators across multiple timeframes
- Market volatility through ATR calculations

### Robust Training Process

- **Double DQN**: Uses separate target and policy networks to reduce overestimation bias
- **Adaptive Learning Rate**: Implements ReduceLROnPlateau scheduler to adjust learning rates based on performance
- **Gradient Clipping**: Prevents exploding gradients during training
- **Huber Loss**: Provides robustness against outliers in the reward distribution

## Dataset

The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1EjlEv57XID0stzUhedlZh-xRQgcdj6Xi?usp=drive_link). You need to request access from the creator before using the dataset.

### Data Structure

The dataset consists of:
- ETF data files in CSV format
- Stock data files in CSV format
- Metadata for symbols

The training pipeline expects the data to be split into three parts:
- Training set (`df_train.csv`)
- Validation set (`df_validate.csv`)
- Test set (`df_test.csv`)

A utility script `train_test_split.py` is provided to help with dataset preparation.

## File Structure

<img width="450" alt="image" src="https://github.com/user-attachments/assets/d4dcbda7-d3b2-4304-88c9-7e5bc76e3637" />

## Implementation Details

### Configuration

The model uses a configuration class that controls various hyperparameters:
- Learning rate: 0.0005
- Batch size: 32 (recommended to increase to 64 or 128)
- Gamma (discount factor): 0.95
- Epsilon parameters for exploration
- Memory size: 50,000 experiences
- Prioritized replay parameters (alpha, beta)

### Technical Indicators

The model automatically calculates and uses the following indicators:
- **Basic price data**: Open, High, Low, Close, Volume
- **RSI (Relative Strength Index)**: Measures the speed and change of price movements
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **ATR (Average True Range)**: Measures market volatility

### Training Process

The training loop:
1. Initializes the environment with training data
2. Creates a DQN agent with policy and target networks
3. For each episode:
   - Resets the environment
   - Iteratively selects actions, observes rewards, and stores transitions
   - Periodically updates the policy network using prioritized experience replay
   - Updates the target network every N steps
   - Validates performance on validation data
   - Saves the best model based on validation returns
4. Plots training metrics including returns, loss, and Q-value errors

### Prediction

The prediction function:
1. Loads a trained model
2. Initializes the environment with test data
3. Runs the model to generate trading signals
4. Calculates portfolio performance
5. Generates visualizations of trading signals, portfolio value, and daily returns

## Usage Instructions

### Prerequisites

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Matplotlib
- scikit-learn

### Setup

1. Clone this repository
2. Download the dataset from the provided Google Drive link
3. Place the dataset in the appropriate directory structure
4. Run the `train_test_split.py` script if needed to prepare your data

### Training

Run the training notebook or execute the `high_level.py` script:

```python
from MacroHFT.MacroHFT_miniproject.RL.agent.high_level import train

# Train the model
train()
```

### Making Predictions

After training, you can make predictions using:

```python
from MacroHFT.MacroHFT_miniproject.RL.agent.high_level import predict

# Make predictions using the trained model
results = predict(model_path="best_model.pt")
```

## Performance Considerations

For optimal performance:

- **Batch size**: Increase to 64 or 128 (default is 32)
- **Episodes**: Increase to 1000 for better convergence
- **Dataset**: Use the full dataset for training
- **Hardware**: Use GPU for significantly faster training (CPU training will be slow)

## Results

The repository includes sample results from previous runs in the `results.zip` file, which contains:
- Trained models
- Prediction outputs
- Performance visualizations
- Code snapshots used for each experiment

Note: These results are for reference only. Delete them before starting your own project.

## Research Paper

This implementation is based on the KDD 2024 paper:
["MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading"](https://arxiv.org/abs/2406.14537)

## Acknowledgments

This repository is inspired by the original implementation at [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT.git).

## Technical Documentation

### AI-Enhanced Documentation

This README was created using an advanced AI assistant (OpenHands) that was connected to the GitHub repository through a secure API integration. The technical implementation involved:

- OAuth2 token-based authentication with GitHub's REST API
- Secure webhook integration for real-time repository event processing
- Custom Git operations through programmatic interfaces rather than CLI commands
- Automated repository cloning and analysis using abstract syntax tree parsing
- Markdown rendering with GitHub-flavored syntax support

The AI analyzed the codebase structure, examined implementation details, and generated comprehensive documentation while preserving critical elements from the original README.
