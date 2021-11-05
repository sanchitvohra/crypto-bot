import talib

def macd(df):
    close = df['close'].to_numpy()
    macd = talib.MACD(close)
    return macd[0]

def rsi(df):
    close = df['close'].to_numpy()
    rsi = talib.RSI(close)
    return rsi

def cci(df):
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    cci = talib.CCI(high, low, close)
    return cci

def adx(df):
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    adx = talib.ADX(high, low, close)
    return adx    
