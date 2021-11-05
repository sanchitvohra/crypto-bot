import numpy as np
import pandas as pd
import random

class CryptoEnv():
    def __init__(self, data, balance, trading_fee, max_action):
        self.data = data
        self.trading_fee = trading_fee
        self.max_action = max_action

        self.time = 0 # random start
        self.portfolio = balance
        self.balance = balance
        self.coins = {'BCH': 0, 'BTC': 1, 'ETH': 2, 'LTC': 3, 'XRP': 4}
        self.account = np.zeros(len(self.coins), dtype=np.float32) # amount of each coin owned

        self.state_index = random.randint(0, data.shape[1] - 1000) # randomly initialize start point
        self.state = self.data[:, self.state_index, :]

    def step(self, actions):
        # amount in USD to buy/sell for each coin
        exec_actions = actions * self.max_action
        for coin in self.coins.keys():
            if exec_actions[self.coins[coin]] < 0:
                self.sell(self.coins[coin], -1 * exec_actions[self.coins[coin]])
            if exec_actions[self.coins[coin]] > 0:
                self.buy(self.coins[coin], exec_actions[self.coins[coin]])

        self.state_index += 1
        self.state = self.data[:, self.state_index, :]        

    def buy(self, coin, amount):
        if amount > self.balance:
            amount = self.balance

        # pessimistically take high price
        account = amount / self.state[coin, 1]
        self.account[coin] += account
        self.balance -= amount

    def sell(self, coin, amount):
        # pessimistically take high price
        account = amount / self.state[coin, 1]
        if account > self.account[coin]:
            account = self.account[coin]

        self.account[coin] -= account
        self.balance += amount