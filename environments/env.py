import numpy as np
import pandas as pd
import random

class CryptoEnv():
    def __init__(self, data, balance, max_trade, trading_fee, history_len):
        self.data = data
        self.trading_fee = trading_fee
        self.history_len = history_len
        self.starting_balance = balance
        self.max_trade = max_trade

        self.coins = {'BCH': 0, 'BTC': 1, 'ETH': 2, 'LTC': 3, 'XRP': 4}

        self.reset()

    def reset(self, state_index = None):
        self.portfolio = self.starting_balance
        if not state_index:
            self.state_index = random.randint(0, self.data.shape[1] - 30001 - self.history_len)
        else:
            self.state_index = state_index
        self.state = self.data[:, self.state_index, :]
        self.prev_states = []

        for i in range(self.history_len):
            self.prev_states.append(self.normalize_state(self.state))
            self.state_index += 1
            self.state = self.data[:, self.state_index, :]
        
        self.balance = self.portfolio
        self.account = np.zeros(len(self.coins), dtype=np.float32)
        self.account_dollars = np.zeros(len(self.coins), dtype=np.float32)

        self.num_coins = len(self.coins)
        self.state_dim = self.state.shape[1]

    def validate(self, index):
        self.reset(self.data.shape[1] - 30001 - self.history_len + index*10000)

    def step(self, actions):
        exec_actions = actions * self.max_trade
        for coin in self.coins.keys():
            if exec_actions[self.coins[coin]] < 0:
                self.sell(self.coins[coin], -1 * exec_actions[self.coins[coin]])
            if exec_actions[self.coins[coin]] > 0:
                self.buy(self.coins[coin], exec_actions[self.coins[coin]])

        if self.history_len > 0:
            self.prev_states = self.prev_states[1:]
            self.prev_states.append(self.normalize_state(self.state))


        self.state_index += 1
        self.state = self.data[:, self.state_index, :]
        self.account_dollars = self.state[:, 1] * self.account
        old_portfolio = self.portfolio
        self.portfolio = self.balance + np.sum(self.state[:, 1] * self.account)
        return self.portfolio - old_portfolio

    def buy(self, coin, amount):
        if amount < 250.0:
            return
        if amount > self.balance:
            amount = self.balance
            
        trading_fee = amount * self.trading_fee
        buy_amount = amount - trading_fee
        buy_quantity = buy_amount / self.state[coin, 1]

        self.account[coin] += buy_quantity
        self.balance -= buy_amount

    def sell(self, coin, amount):
        if amount < 250.0:
            return
        quantity = amount / self.state[coin, 1]

        if quantity > self.account[coin]:
            quantity = self.account[coin]

        sell_amount = quantity * self.state[coin, 1]
        sell_quantity = quantity

        self.account[coin] -= sell_quantity
        self.balance += sell_amount

    def normalize_state(self, state):
        normalized_state = np.copy(state)
        normalized_state[:, :4] = normalized_state[:, :4] * (10 ** -5)
        normalized_state[:, 4] = normalized_state[:, 4] * (10 ** -8)
        normalized_state[:, 5:] = normalized_state[:, 5:] * (10 ** -4)
        return normalized_state

    def get_state(self, flatten):
        normalized_state = self.normalize_state(self.state)
        if self.history_len > 0:
            normalized_history = np.stack(self.prev_states)
        normalized_account = np.copy(self.account_dollars)
        normalized_account = normalized_account * (10 ** -6)
        normalized_balance = np.copy(self.balance)
        normalized_balance = normalized_balance * (10 ** -8)
        
        if flatten:
            normalized_state = normalized_state.reshape(-1)
            normalized_account = np.hstack([normalized_balance, normalized_account]).reshape(-1)
            if self.history_len > 0:
                normalized_history = normalized_history.reshape(-1)
                return np.hstack([normalized_state, normalized_history, normalized_account])
            else:
                return np.hstack([normalized_state, normalized_account])
        else:
            return normalized_state, np.hstack([normalized_balance, normalized_account])

    def get_price_state(self, flatten, normalize):
        price_state = np.copy(self.state[:, :5])
        if normalize:
            price_state[:, :4] *= (10 ** -5)
            price_state[:, 4] *= (10 ** -8)
        if flatten:
            return price_state.reshape(-1)
        else:
            return price_state

    def unormalize_price(self, price, flatten):
        price_state = np.copy(price)
        if flatten:
            price_state = price_state.reshape(self.num_coins, self.state_dim)
        
        price_state[:, :4] *= (10 ** 5)
        price_state[:, 4] *= (10 ** 8)

        return price_state

    def get_account_state(self, normalize):
        normalized_account = np.copy(self.account_dollars)
        normalized_balance = np.copy(self.balance)
        if normalize:
            normalized_account = normalized_account * (10 ** -6)
            normalized_balance = normalized_balance * (10 ** -8)

        return np.hstack([normalized_balance, normalized_account])

