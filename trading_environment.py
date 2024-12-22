import numpy as np
import pandas as pd
import random
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 stock_dim,
                 state_space,
                 action_space,
                 tech_indicator_list,
                 macro_variable_list,
                 hmax=100,
                 initial_amount=1000000,
                 buy_cost_pct=0.001,
                 sell_cost_pct=0.001,
                 reward_scaling=1e-4,
                 make_plots=False,
                 print_verbosity=1000,
                 day=0,
                 initial=True,
                 previous_state=[],
                 dividend_col='dividends',
                 threshold_holding=0.02):
        self.day = day
        self.df = df
        self.stock_dim = len(self.df.tic.unique())
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = action_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.macro_variable_list = macro_variable_list if macro_variable_list is not None else []
        self.dividend_col = dividend_col
        self.threshold_holding = threshold_holding

        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            # Check that the price is greater than 0 and the action is negative
            if self.state[index + 1] > 0 and action < 0:
                # Only sell if you have shares to sell
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                    sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct)

                    # Update balance with the profit from the sale
                    self.state[0] += sell_amount

                    # Update the state with the number of shares sold
                    self.state[index + self.stock_dim + 1] -= sell_num_shares

                    # Update transaction cost and number of trades made
                    self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                    self.trades += 1
                else:
                    sell_num_shares = 0  # No sale if no shares are available
            else:
                sell_num_shares = 0  # No sale if conditions are not met

            return sell_num_shares

        # Execute the sale
        sell_num_shares = _do_sell_normal()

        return sell_num_shares


    def _buy_stock(self, index, action):

        def _do_buy():
            # Ensure that the price is greater than 0 and the action is not negative
            if self.state[index + 1] > 0 and action > 0:
                # Calculate the number of shares you can buy with the available balance
                available_amount = self.state[0] // self.state[index + 1]

                # Determine how many shares to buy (minimum between what is possible and the action)
                buy_num_shares = min(available_amount, action)

                # Calculate the total amount to pay including the commission
                buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)

                # Deduct the balance for the purchase
                self.state[0] -= buy_amount

                # Check that the balance is not negative
                if self.state[0] < 0:
                    self.state[0] += buy_amount
                    buy_num_shares -= 1
                    buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.state[0] -= buy_amount

                # Update the state with the number of shares bought
                self.state[index + self.stock_dim + 1] += buy_num_shares

                # Update costs and the number of transactions made
                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0  # If conditions are not met, no shares are bought

            return buy_num_shares

        # Execute the purchase
        buy_num_shares = _do_buy()

        return buy_num_shares


    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # End of episode: analysis and metrics
            if self.make_plots:
                self._make_plot()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))

            tot_reward = end_total_asset - self.initial_amount

            # Calculate Sharpe ratio
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()

            # Rewards
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]

            # Print episode statistics
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # Scale actions
            actions = actions * self.hmax
            actions = actions.astype(int)

            # Get current prices from self.data
            current_prices = self.data['close'].values.tolist()

            # Get previous prices from the DataFrame self.df
            if self.day > 0:
                previous_data = self.df.loc[self.day - 1, :]
                previous_prices = previous_data['close'].values.tolist()
            else:
                previous_prices = current_prices  # If it's the first day, use current prices as a reference

            # Initialize penalty for missed opportunity
            hold_penalty = 0

            for i, action in enumerate(actions):
                price_change = (current_prices[i] - previous_prices[i]) / previous_prices[i]

                # Determine transaction cost for buying or selling
                transaction_cost = self.buy_cost_pct if action > 0 else self.sell_cost_pct

                # Decision to act: compare with transaction cost and threshold
                if abs(price_change) <= max(transaction_cost, self.threshold_holding):
                    actions[i] = 0  # Hold position if not profitable to cover the cost or if it doesn't exceed the threshold

                # Penalty for "missed opportunity" if holding and the price change is significant
                if actions[i] == 0 and abs(price_change) > self.threshold_holding:
                    # Calculate penalty for not acting
                    hold_penalty -= (abs(price_change) - transaction_cost) * 10  # Proportional penalty

            # Execute sell actions
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            # Execute buy actions
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            # Store executed actions
            self.actions_memory.append(actions)

            # Move to the next day
            self.day += 1
            self.data = self.df.loc[self.day, :]

            # Update state
            self.state = self._update_state()

            # Calculate total portfolio value including dividends
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])) + \
                              sum(np.array(self.data[self.dividend_col].values.tolist()) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))

            # Calculate reward
            begin_total_asset = self.asset_memory[-1] if self.asset_memory else end_total_asset
            self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling

            # Scale and accumulate penalty for missed opportunity
            self.reward += hold_penalty * self.reward_scaling

            # Store portfolio value
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

        return self.state, self.reward, self.terminal, {}


    def reset(self):

        # Initialize the state (initial capital, prices, positions, technical indicators, etc.)
        self.state = self._initiate_state()

        # Portfolio value memory (reset or continue from previous state)
        if self.initial:
            self.asset_memory = [self.initial_amount]  # Start with the initial capital
        else:
            previous_total_asset = self.previous_state[0] + \
                                  sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                      self.previous_state[(self.stock_dim + 1):(2 * self.stock_dim + 1)]))
            self.asset_memory = [previous_total_asset]  # Continue from the previous state

        # Reset day, data, and other key parameters
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0  # Transaction costs
        self.trades = 0  # Number of transactions
        self.terminal = False  # End of episode indicator

        # Reset reward, action, and date memories
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]  # Store the initial date

        # Increment the episode counter
        self.episode += 1

        # Return the initial state for the agent to start the new episode
        return np.array(self.state)


    def render(self, mode='human', close=False):
        if close:
            return

        # Basic portfolio information
        print(f"Day: {self.day}, Date: {self._get_date()}")
        print(f"Cash available: {self.state[0]:0.2f}")

        # Value of positions in each ETF
        for i in range(self.stock_dim):
            print(f"ETF {i + 1} - Price: {self.state[i + 1]:0.2f}, Shares held: {self.state[self.stock_dim + 1 + i]:0.2f}")

        # Total portfolio value
        portfolio_value = self.state[0] + \
                          sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                              self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
        print(f"Portfolio value: {portfolio_value:0.2f}")

        # Accumulated transaction costs and number of trades
        print(f"Total cost of transactions: {self.cost:0.2f}")
        print(f"Total trades made: {self.trades}")

        # Accumulated rewards
        total_rewards = sum(self.rewards_memory)
        print(f"Total rewards: {total_rewards:0.2f}")
        print("==========================================")


    def _initiate_state(self):
        """
        Initializes the environment state.
        The state contains:
        - Available capital
        - ETF prices
        - ETF positions
        - Technical indicators for each ETF
        - Macroeconomic variables (common for all ETFs)
        """
        if self.initial:
            if self.stock_dim > 1:
                # For multiple ETFs

                # ETF prices
                etf_prices = self.data['close'].values.tolist()

                # Initial positions (0 shares for each ETF)
                etf_positions = [0] * self.stock_dim

                # Technical indicators specific to each ETF
                tech_indicators = []
                for tech in self.tech_indicator_list:
                    tech_indicators.extend(self.data[tech].values.tolist())  # Add indicators for each ETF

                # Common macroeconomic variables (only once, not repeated per ETF)
                macro_variables = [self.data[var].values[0] for var in self.macro_variable_list]

                # Create the state with capital, prices, positions, technical indicators, and macroeconomic variables
                state = [self.initial_amount] + etf_prices + etf_positions + tech_indicators + macro_variables
            else:
                # For a single ETF
                etf_price = [self.data['close'].values[0]]
                etf_position = [0]  # Only 1 position for one ETF

                # Technical indicators for the single ETF
                tech_indicators = [self.data[tech].values[0] for tech in self.tech_indicator_list]

                # Common macroeconomic variables (not repeated per ETF)
                macro_variables = [self.data[var].values[0] for var in self.macro_variable_list]

                # Create the state with capital, price, position, technical indicators, and macroeconomic variables
                state = [self.initial_amount] + etf_price + etf_position + tech_indicators + macro_variables
        else:
            # Case where state is restored from previous state
            if self.stock_dim > 1:
                etf_prices = self.data['close'].values.tolist()
                etf_positions = self.previous_state[(self.stock_dim + 1):(2 * self.stock_dim + 1)]

                tech_indicators = []
                for tech in self.tech_indicator_list:
                    tech_indicators.extend(self.data[tech].values.tolist())

                macro_variables = [self.data[var].values[0] for var in self.macro_variable_list]

                state = [self.previous_state[0]] + etf_prices + etf_positions + tech_indicators + macro_variables
            else:
                etf_price = [self.data['close'].values[0]]
                etf_position = self.previous_state[(self.stock_dim + 1):(2 * self.stock_dim + 1)]

                tech_indicators = [self.data[tech].values[0] for tech in self.tech_indicator_list]
                macro_variables = [self.data[var].values[0] for var in self.macro_variable_list]

                state = [self.previous_state[0]] + etf_price + etf_position + tech_indicators + macro_variables

        return state


    def _update_state(self):
        """
        Updates the environment state, maintaining the same structure as the initial state.
        The state contains:
        - Available capital
        - ETF prices
        - ETF positions
        - Technical indicators for each ETF
        - Common macroeconomic variables
        """
        if self.stock_dim > 1:
            # Updated ETF prices
            etf_prices = self.data['close'].values.tolist()

            # Current ETF positions
            etf_positions = self.state[(1 + self.stock_dim):(1 + 2 * self.stock_dim)]  # Keep previous positions

            # Updated technical indicators for each ETF
            tech_indicators = []
            for tech in self.tech_indicator_list:
                tech_indicators.extend(self.data[tech].values.tolist())

            # Updated macroeconomic variables (only once, not per ETF)
            macro_variables = [self.data[var].values[0] for var in self.macro_variable_list]

            # Create the new state with capital, prices, positions, technical indicators, and macroeconomic variables
            state = [self.state[0]] + etf_prices + etf_positions + tech_indicators + macro_variables
        else:
            # For a single ETF
            etf_price = [self.data['close'].values[0]]
            etf_position = self.state[(1 + self.stock_dim):(1 + 2 * self.stock_dim)]  # Keep previous position

            # Updated technical indicators
            tech_indicators = [self.data[tech].values[0] for tech in self.tech_indicator_list]

            # Updated macroeconomic variables (common, not repeated per ETF)
            macro_variables = [self.data[var].values[0] for var in self.macro_variable_list]

            # Create the new state with capital, price, position, technical indicators, and macroeconomic variables
            state = [self.state[0]] + etf_price + etf_position + tech_indicators + macro_variables

        return state


    def _get_date(self):
        # If working with multiple ETFs
        if len(self.df.tic.unique()) > 1:
            # Get the unique date of the current day
            date = self.data['datadate'].unique()[0]
        else:
            # If working with a single ETF, get the date directly
            date = self.data['datadate']

        return date


    def _make_plot(self):
        """
        Generates a plot of the portfolio value evolution during the episode.
        """
        plt.figure(figsize=(10, 6))

        # Plot portfolio value (account_value) over the episode
        plt.plot(self.asset_memory, label='Portfolio Value')

        # Labels and title
        plt.title('Portfolio Value Evolution')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')

        plt.legend()
        plt.show()


    def save_asset_memory(self):
        # This function is used to save the portfolio evolution
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({'date': date_list, 'account_value': asset_list})
        return df_account_value


    def save_action_memory(self):
        # This function is used to save the agent's actions
        # For multiple ETFs
        if len(self.df.tic.unique()) > 1:
            # List of dates (excluding the last to align with actions)
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory

            df_date = pd.DataFrame(date_list, columns=['date'])
            df_actions = pd.DataFrame(action_list, columns=self.data.tic.values)

            # Align actions with dates
            df_actions.index = df_date['date']
        else:
            # For a single ETF
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory

            df_actions = pd.DataFrame({'date': date_list, 'actions': action_list})

        return df_actions


    def _seed(self, seed=None):
        # Seed the random number generator for gym
        self.np_random, seed = seeding.np_random(seed)
        # Ensure the seed is valid (an integer in the range 0 <= seed < 2^32)
        seed = int(seed) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]