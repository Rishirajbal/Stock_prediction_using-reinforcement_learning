import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ====================== Configuration ======================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    lr = 0.0005
    batch_size = 32
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update = 10
    memory_size = 50000
    num_episodes = int(input("Enter number of episodes: "))
    alpha = 0.6  # For prioritized replay
    beta_start = 0.4  # For prioritized replay
    beta_frames = 100000  # Annealing rate for beta
    
    # Data paths
    base_path = r"C:\Users\KIIT\OneDrive\Desktop\MacroHFT\MacroHFT_miniproject"
    train_path = os.path.join(base_path, "datasets", "df_train.csv")
    val_path = os.path.join(base_path, "datasets", "df_validate.csv")
    test_path = os.path.join(base_path, "datasets", "df_test.csv")
    
    # Dataset limitation
    max_train_rows = int(input("Enter number of max train rows: "))
    max_val_rows = int(input("Enter number of max val rows: "))
    
    # Basic indicators
    base_indicators = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Advanced indicators
    advanced_indicators = ['RSI', 'MACD', 'ATR']
    
    # Company/ticker information
    ticker_column = 'Ticker'
    
    # Features to use (will be set after data loading)
    tech_indicators = []

# Set random seeds
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)

# Define experience tuple
Experience = namedtuple('Experience', 
                       ('state', 'action', 'reward', 'next_state', 'done'))

# ====================== Neural Network ======================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ====================== Robust Prioritized Replay Buffer ======================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        self.beta = Config.beta_start
        self.beta_increment = (1.0 - Config.beta_start) / Config.beta_frames
        
    def add(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(*args))
        else:
            self.buffer[self.pos] = Experience(*args)
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
            
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get current priorities
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:self.pos]
        else:
            priorities = self.priorities
        
        # Calculate probabilities with numerical stability
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        # Handle cases where probabilities might be invalid
        if probs_sum <= 0 or np.isnan(probs_sum):
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = probs / probs_sum
        
        # Sample indices
        try:
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        except ValueError:
            # Fallback to uniform sampling if there's an issue
            indices = np.random.choice(len(self.buffer), batch_size)
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        # Clip priorities to avoid extreme values
        priorities = np.clip(priorities, 1e-5, 1e5)
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        self.max_priority = max(self.max_priority, np.max(priorities))

# ====================== Technical Indicator Calculations ======================
def calculate_technical_indicators(df):
    """Calculate technical indicators if they don't exist in the dataframe"""
    df = df.copy()
    
    # Calculate RSI if not present
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD if not present
    if 'MACD' not in df.columns:
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
    
    # Calculate ATR if not present
    if 'ATR' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
    
    return df.dropna()

# ====================== Trading Environment ======================
class TradingEnv:
    def __init__(self, df, initial_balance=10000, transaction_cost=0.0002):
        print("\n" + "="*50)
        print("Initializing environment...")
        
        # Store ticker information if available
        self.tickers = df[Config.ticker_column].unique() if Config.ticker_column in df.columns else ['Unknown']
        
        # Calculate technical indicators
        self.df = calculate_technical_indicators(df[Config.base_indicators + ([Config.ticker_column] if Config.ticker_column in df.columns else [])])
        
        # Set the indicators we'll actually use
        Config.tech_indicators = Config.base_indicators + [
            ind for ind in Config.advanced_indicators if ind in self.df.columns
        ]
        print(f"Using indicators: {Config.tech_indicators}")
        
        # Normalize data
        self.scaler = StandardScaler()
        self.df[Config.tech_indicators] = self.scaler.fit_transform(self.df[Config.tech_indicators])
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.action_space = [0, 1]
        self.reset()
        
        print(f"Environment initialized with {len(self.df)} timesteps")
        if len(self.tickers) > 1:
            print(f"Tracking {len(self.tickers)} tickers: {', '.join(self.tickers)}")
        print("="*50 + "\n")
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0
        self.portfolio_value = [self.initial_balance]
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        state = self.df.iloc[self.current_step][Config.tech_indicators].values.astype(np.float32)
        # Add position information
        position = np.array([self.holding > 0], dtype=np.float32)
        state = np.concatenate([state, position])
        return torch.FloatTensor(state).to(Config.device)
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        current_ticker = self.df.iloc[self.current_step][Config.ticker_column] if Config.ticker_column in self.df.columns else 'Unknown'
        
        # Execute action
        if action == 1 and self.holding == 0:  # Buy
            cost = current_price * (1 + self.transaction_cost)
            self.holding = self.balance / cost
            self.balance = 0
            self.trades.append({
                'step': self.current_step,
                'ticker': current_ticker,
                'action': 'buy',
                'price': current_price,
                'shares': self.holding
            })
        
        elif action == 0 and self.holding > 0:  # Sell
            self.balance = self.holding * current_price * (1 - self.transaction_cost)
            self.trades.append({
                'step': self.current_step,
                'ticker': current_ticker,
                'action': 'sell',
                'price': current_price,
                'shares': self.holding,
                'proceeds': self.balance
            })
            self.holding = 0
        
        # Update portfolio
        new_value = self.balance + (self.holding * next_price)
        reward = np.log(new_value / self.portfolio_value[-1]) * 100 if self.portfolio_value[-1] > 0 else 0
        self.portfolio_value.append(new_value)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'portfolio_value': new_value,
            'position': 'long' if self.holding > 0 else 'cash',
            'ticker': current_ticker
        }
        
        return self._get_state(), reward, done, info

# ====================== RL Agent ======================
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        print("Initializing DQN agent with:")
        print(f"• Input dim: {input_dim} (features + position)")
        print(f"• Output dim: {output_dim} (actions)")
        
        self.policy_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.memory = PrioritizedReplayBuffer(Config.memory_size, Config.alpha)
        self.epsilon = Config.epsilon_start
        self.loss_fn = nn.HuberLoss()
        self.steps_done = 0
        
        print(f"Agent initialized on {Config.device}")
        
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def update_model(self):
        if len(self.memory.buffer) < Config.batch_size:
            return 0, 0
        
        try:
            # Sample from prioritized replay
            batch, indices, weights = self.memory.sample(Config.batch_size)
            weights = torch.FloatTensor(weights).to(Config.device)
            
            # Unpack batch
            states = torch.stack([exp.state for exp in batch])
            actions = torch.LongTensor([exp.action for exp in batch]).to(Config.device)
            rewards = torch.FloatTensor([exp.reward for exp in batch]).to(Config.device)
            next_states = torch.stack([exp.next_state for exp in batch])
            dones = torch.FloatTensor([exp.done for exp in batch]).to(Config.device)
            
            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Double DQN target calculation
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                target_q = rewards + (1 - dones) * Config.gamma * next_q
            
            # Compute loss with importance sampling weights
            loss = (weights * self.loss_fn(current_q.squeeze(), target_q)).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities with absolute TD errors
            errors = (current_q.squeeze() - target_q).abs().cpu().detach().numpy()
            self.memory.update_priorities(indices, errors + 1e-5)
            
            # Decay epsilon
            self.epsilon = max(Config.epsilon_end, self.epsilon * Config.epsilon_decay)
            
            return loss.item(), errors.mean()
            
        except Exception as e:
            print(f"Error in update_model: {str(e)}")
            return 0, 0
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ====================== Training Loop ======================
def train():
    print("="*50)
    print("Starting training process...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # Load data
    print("Loading training data...")
    try:
        train_df = pd.read_csv(Config.train_path)
        val_df = pd.read_csv(Config.val_path)
        
        if Config.max_train_rows:
            train_df = train_df.iloc[:Config.max_train_rows]
        if Config.max_val_rows:
            val_df = val_df.iloc[:Config.max_val_rows]
            
        print(f"Training data shape: {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")
        
        # Check for ticker information
        if Config.ticker_column in train_df.columns:
            print(f"\nFound ticker information. Unique tickers:")
            print(train_df[Config.ticker_column].value_counts())
    except Exception as e:
        print(f" Error loading data: {str(e)}")
        return
    
    # Initialize environment and agent
    print("\nInitializing environment...")
    env = TradingEnv(train_df)
    agent = DQNAgent(input_dim=len(Config.tech_indicators)+1, output_dim=2)
    
    # Training metrics
    best_val_return = -np.inf
    returns = []
    losses = []
    q_values = []
    
    print("\n" + "="*50)
    print("Beginning training...")
    print("="*50 + "\n")
    
    for episode in range(Config.num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        episode_errors = 0
        update_count = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            loss, error = agent.update_model()
            if loss > 0:
                episode_loss += loss
                episode_errors += error
                update_count += 1
            
            state = next_state
            total_reward += reward
        
        # Validation
        val_env = TradingEnv(val_df)
        val_state = val_env.reset()
        val_done = False
        val_return = 0
        
        with torch.no_grad():
            while not val_done:
                val_action = agent.policy_net(val_state).argmax().item()
                val_state, val_reward, val_done, val_info = val_env.step(val_action)
                val_return += val_reward
        
        # Calculate metrics
        avg_loss = episode_loss / max(1, update_count)
        avg_error = episode_errors / max(1, update_count)
        
        returns.append(val_return)
        losses.append(avg_loss)
        q_values.append(avg_error)
        
        # Save best model
        if val_return > best_val_return:
            best_val_return = val_return
            # Save only the state dict for more reliable loading
            torch.save(agent.policy_net.state_dict(), "best_model.pt")
            print(f"Saved new best model with return: {best_val_return:.2f}")
        
        # Update target network
        if episode % Config.target_update == 0:
            agent.update_target()
        
        # Adjust learning rate
        agent.scheduler.step(avg_loss)
        
        # Print progress
        print(f"Ep {episode+1:03d}/{Config.num_episodes} | "
              f"Train R: {total_reward:+.2f} | "
              f"Val R: {val_return:+.2f} | "
              f"ε: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Q Error: {avg_error:.4f} | "
              f"LR: {agent.optimizer.param_groups[0]['lr']:.2e} | "
              f"Ticker: {info.get('ticker', 'N/A')}")
    
    # Training complete
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation return: {best_val_return:.2f}")
    print("="*50 + "\n")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(returns)
    plt.title("Validation Returns")
    plt.ylabel("Return")
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(q_values)
    plt.title("Q-value Error")
    plt.ylabel("Error")
    plt.xlabel("Episode")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

# ====================== Prediction Function ======================
def predict(model_path="best_model.pt", test_data_path=None):
    print("\n" + "="*50)
    print("Starting prediction...")
    
    # Load test data
    test_path = test_data_path if test_data_path else Config.test_path
    try:
        test_df = pd.read_csv(test_path)
        print(f"Loaded test data with {len(test_df)} rows")
        
        if Config.ticker_column in test_df.columns:
            print("Tickers in test data:")
            print(test_df[Config.ticker_column].value_counts())
    except Exception as e:
        print(f"❌ Error loading test data: {str(e)}")
        return None
    
    # Initialize environment
    env = TradingEnv(test_df)
    
    # Initialize agent
    agent = DQNAgent(input_dim=len(Config.tech_indicators)+1, output_dim=2)
    
    # Load trained model state dict
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=Config.device))
        agent.policy_net.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None
    
    # Run prediction
    state = env.reset()
    done = False
    actions = []
    portfolio_values = []
    trade_history = []
    
    with torch.no_grad():
        while not done:
            action = agent.policy_net(state).argmax().item()
            state, _, done, info = env.step(action)
            actions.append(action)
            portfolio_values.append(info['portfolio_value'])
            if 'ticker' in info:
                trade_history.append(info['ticker'])
    
    # Create results dataframe
    results = test_df.iloc[:len(actions)].copy()
    results['Action'] = actions
    results['Portfolio_Value'] = portfolio_values
    results['Daily_Return'] = results['Portfolio_Value'].pct_change()
    
    if trade_history:
        results['Ticker'] = trade_history
    
    # Save predictions
    os.makedirs('./results', exist_ok=True)
    results.to_csv('./results/predictions.csv', index=False)
    print("\nPrediction results saved to ./results/predictions.csv")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Price and actions
    plt.subplot(3, 1, 1)
    plt.plot(results['Close'], label='Price')
    
    buy_signals = results[results['Action'] == 1]
    sell_signals = results[results['Action'] == 0]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], 
                color='green', label='Buy', marker='^', alpha=0.7)
    plt.scatter(sell_signals.index, sell_signals['Close'], 
                color='red', label='Sell', marker='v', alpha=0.7)
    
    plt.title('Trading Signals')
    plt.legend()
    plt.grid()
    
    # Portfolio value
    plt.subplot(3, 1, 2)
    plt.plot(results['Portfolio_Value'], label='Portfolio Value')
    plt.title('Portfolio Performance')
    plt.legend()
    plt.grid()
    
    # Daily returns
    plt.subplot(3, 1, 3)
    plt.bar(results.index, results['Daily_Return'], 
            color=np.where(results['Daily_Return'] > 0, 'g', 'r'))
    plt.title('Daily Returns')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('./results/prediction_results.png')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Prediction Summary:")
    print(f"Initial Portfolio Value: ${results['Portfolio_Value'].iloc[0]:,.2f}")
    print(f"Final Portfolio Value: ${results['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(results['Portfolio_Value'].iloc[-1]/results['Portfolio_Value'].iloc[0]-1)*100:.2f}%")
    print(f"Number of Trades: {len(buy_signals) + len(sell_signals)}")
    
    if 'Ticker' in results.columns:
        print("\nTrades by Ticker:")
        print(results.groupby('Ticker')['Action'].value_counts())
    print("="*50 + "\n")
    
    return results

if __name__ == "__main__":
    # First train the model
    train()
    
    # Then make predictions
    predict()