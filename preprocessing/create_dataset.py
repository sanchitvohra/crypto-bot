import pandas as pd
import numpy as np
from preprocessing.indicators import *

coins = ['BCH', 'BTC', 'ETH', 'LTC', 'XRP']
coins_idx = {'BCH': 0, 'BTC': 1, 'ETH': 2, 'LTC': 3, 'XRP': 4}

def process_one(coin):
    df = pd.read_csv("./data/Bitstamp_" + coin + "USD.csv", low_memory=False)
    data_macd = macd(df)
    data_rsi = rsi(df)
    data_cci = cci(df)
    data_adx = adx(df)
    
    df = df.assign(MACD = data_macd)
    df = df.assign(RSI = data_rsi)
    df = df.assign(CCI = data_cci)
    df = df.assign(ADX = data_adx)
    return df[['open', 'high', 'low', 'close', 'Volume USD', 'RSI', 'MACD', 'CCI', 'ADX']].to_numpy()

def process_all():
    frames = []
    sizes = []
    for coin in coins:
        df = process_one(coin)
        frames.append(df)
        sizes.append(df.shape[0])

    min_size = min(sizes)

    for i in range(len(frames)):
        frames[i] = frames[i][:min_size, :]

    combined = np.stack(frames)
    idx = np.argwhere(np.isnan(combined))
    first_defined = np.max(idx[:, 1]) + 1
    final = combined[:, first_defined:, :].astype(np.float32)
    max_norm = np.max(final, axis=1)
    min_norm = np.min(final, axis=1)
    return final, max_norm, min_norm