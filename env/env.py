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

        self.portfolio = balance
        self.balance = balance
        self.coins = {'BCH': 0, 'BTC': 1, 'ETH': 2, 'LTC': 3, 'XRP': 4}

        self.reset()

    def reset(self):
        self.portfolio = self.starting_balance
        self.state_index = random.randint(0, self.data.shape[1] - 1000)
        self.state = self.data[:, self.state_index, :]

        weights = np.random.dirichlet(np.ones(1 + len(self.coins)), size=1)[0]
        
        self.balance = weights[0] * self.portfolio
        self.account = np.zeros(len(self.coins), dtype=np.float32)

        for coin in range(len(self.coins)):
            self.account[coin] = (weights[coin + 1] * self.portfolio) / self.state[coin, 1] 

    def step(self, actions):
        exec_actions = actions * self.max_trade
        for coin in self.coins.keys():
            if exec_actions[self.coins[coin]] < 0:
                self.sell(self.coins[coin], -1 * exec_actions[self.coins[coin]])
            if exec_actions[self.coins[coin]] > 0:
                self.buy(self.coins[coin], exec_actions[self.coins[coin]])

        self.state_index += 1
        self.state = self.data[:, self.state_index, :]        

    def buy(self, coin, quantity):
        amount = quantity * self.state[coin, 1]

        if amount > self.balance:
            amount = self.balance
            
        buy_amount = (1 - self.trading_fee) * amount    
        buy_quantity = amount / self.state[coin, 1]

        self.account[coin] += buy_quantity
        self.balance -= buy_amount

    def sell(self, coin, quantity):
        if quantity > self.account[coin]:
            quantity = self.account[coin]

        sell_amount = quantity * self.state[coin, 1]
        sell_quantity = quantity

        self.account[coin] -= sell_quantity
        self.balance += sell_amount


    def get_state(self):
        normalized_state = self.state
        normalized_state[:, :4] = normalized_state[:, :4] * (10 ** -5)
        normalized_state[:, 4] = normalized_state[:, 4] * (10 ** -8)
        normalized_state[:, 5:] = normalized_state[:, 5:] * (10 ** -4)
        normalized_account = self.account
        normalized_account = normalized_account / (self.max_trade * (10 ** 4))
        normalized_balance = self.balance
        normalized_balance = normalized_balance / (self.starting_balance * (10 ** 3))
        
        return normalized_state, np.hstack([normalized_balance, normalized_account])