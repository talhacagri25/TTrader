# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:31:48 2023

@author: talha
"""

import ccxt
import pandas as pd
import time
import pytz
from decimal import Decimal
from datetime import datetime
import numpy as np

api_key = 'xxxxxxx'
secret_key = 'xxxxxxx'

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

symbol = 'BTC/USDT'  # symbol to trade
timeframe='1m'
leverage = 10  # leverage
previous_state = None  # keep track of the previous state

mafast = 1
maslow = 2
masafety = 55
ma_htf = 9
matype='EMA'
higher_timeframe1 = '1s'
params1 = {
    'price': 'mark'
}

def convert_to_utc_plus_3(timestamp):
    utc = pytz.UTC
    utc_plus_3 = pytz.timezone('Etc/GMT-3')
    localized_timestamp = utc.localize(timestamp)
    return localized_timestamp.astimezone(utc_plus_3)

def print_signal(signal):
    local_timestamp = convert_to_utc_plus_3(signal['timestamp'])
    time_str = local_timestamp.strftime('%H:%M:%S')
    print(f"Timestamp: {time_str}, Signal: {signal['signal']}")
    
def get_historical_data(timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    data = []
    for o in ohlcv:
        data.append([o[0], o[1], o[2], o[3], o[4], o[5]])
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_moving_average(matype, data, window):
    if matype == 'SMA':
        return data.rolling(window).mean()
    elif matype == 'EMA':
        return data.ewm(span=window).mean()
    elif matype == 'WMA':
        weights = np.arange(1, window+1)
        return data.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    else:
        return None

def detect_signals(df):
    df['crs_buy'] = df['avgma_fast'].gt(df['avgma_safety'].shift(1)) & df['avgma_fast'].shift(1).le(df['avgma_safety'].shift(1))
    df['crs_sell'] = df['avgma_fast'].lt(df['avgma_safety'].shift(1)) & df['avgma_fast'].shift(1).ge(df['avgma_safety'].shift(1))
    return df

def resample_to_higher_timeframe(df, higher_timeframe1):
    df = df.set_index('timestamp')
    df = df.resample(higher_timeframe1).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df = df.dropna()
    df = df.reset_index()
    return df

def calculate_htf_moving_average(df, ma_htf):
    df['avgma_htf'] = df['close'].ewm(span=ma_htf).mean()
    return df

def calculate_safety_zones(df):
    df['safety_zone'] = df['avgma_fast'].ge(df['avgma_slow']) & df['avgma_slow'].ge(df['avgma_safety'])
    df['unsafety_zone'] = df['avgma_fast'].le(df['avgma_slow']) & df['avgma_slow'].le(df['avgma_safety'])
    return df

def set_leverage(leverage):
    params = {'symbol': 'BTCUSDT', 'leverage': leverage}
    return exchange.fapiPrivate_post_leverage(params)

def create_order(symbol, side, amount, price=None):
    params = {'quoteOrderQty': amount}
    try:
        print(f"Creating {side} order with amount {amount}")
        exchange.create_order(symbol, 'market', side, str(amount), price, params)
        trades = exchange.fetch_my_trades(symbol)
        for trade in trades[-1:]:  # print the 5 most recent trades
            print(trade)
    except ccxt.InsufficientFunds as e:
        print('Insufficient funds', e)
    except Exception as e:
        print('An error occurred', e)

def get_balance():
    balance = exchange.fetch_balance()
    free_usdt = balance['free']['USDT']
    return free_usdt

def get_position():
    try:
        # replace 'BTC/USDT' with your actual trading symbol
        positions = exchange.fetch_positions(['BTC/USDT:USDT'])
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                return position
        return None
    except Exception as e:
        print("Error fetching position: ", e)
        return None

def get_fee_rate():
    fee = exchange.fetch_trading_fees()
    return str(fee['BTC/USDT:USDT']['taker'])

def get_ticker_price():
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    return price

def adjust_for_fee_rate_and_price(amount):
    print(f"Adjusting amount {amount} for fee and price")
    amount = Decimal(amount)
    fee_rate = Decimal(get_fee_rate())
    ticker_price = Decimal(get_ticker_price())
    one = Decimal(1)
    adjusted_amount = (amount / ticker_price) * (one - fee_rate)
    return str(adjusted_amount.quantize(Decimal("1.00000")))

def trading_logic(current_state):
    global previous_state
    global latest_signal
    global latest_htf_signal

    if current_state == 'Safety Zone' and previous_state in ['Neutral Zone', 'Unsafety Zone'] :
        balance = get_balance()
        balance = float(adjust_for_fee_rate_and_price(balance))*0.99*leverage
        create_order('BTC/USDT:USDT','buy', balance)

    elif current_state == 'Neutral Zone' and previous_state == 'Safety Zone':
        position = get_position()
        if position:
            sell_qty = float(position['info']['positionAmt']) * 0.75
            create_order('BTC/USDT:USDT','sell', sell_qty)

    elif current_state == 'Unsafety Zone' and previous_state in ['Safety Zone', 'Neutral Zone']:
        position = get_position()
        if position:
            sell_qty = position['info']['positionAmt']
            create_order('BTC/USDT:USDT','sell', sell_qty)

    elif current_state == 'Neutral Zone' and previous_state == 'Unsafety Zone':
        balance = get_balance()
        buy_qty = balance * 0.25 * leverage
        buy_qty = adjust_for_fee_rate_and_price(buy_qty)
        create_order('BTC/USDT:USDT', 'buy', buy_qty)

    previous_state = current_state


def main():
    global latest_signal, latest_htf_signal
    latest_signal = None
    latest_htf_signal = None
    set_leverage(leverage)
    retries = 0
    max_retries = 5
    while True:
        try:
            current_time = exchange.fetch_time()
            current_time = datetime.utcfromtimestamp(current_time / 1000)  # convert from milliseconds to datetime
            if current_time.second < 5:  # start of a new minute, with a 5-second buffer
                df = get_historical_data(timeframe)
                df['avgma_fast'] = calculate_moving_average(matype, df['close'], mafast)
                df['avgma_slow'] = calculate_moving_average(matype, df['close'], maslow)
                df['avgma_safety'] = calculate_moving_average(matype, df['close'], masafety)
                df = detect_signals(df)
                df_htf = resample_to_higher_timeframe(df, higher_timeframe1)
                df_htf['avgma_htf'] = calculate_moving_average(matype, df_htf['close'], ma_htf)
                df = calculate_safety_zones(df)
                latest_signal = df.iloc[-1]
                latest_htf_signal = df_htf.iloc[-1]
                safety_zone = latest_signal['safety_zone']
                unsafety_zone = latest_signal['unsafety_zone']

                if safety_zone:
                    current_state = "Safety Zone"
                elif unsafety_zone:
                    current_state = "Unsafety Zone"
                else:
                    current_state = "Neutral Zone"
                    
                trading_logic(current_state)
                if (latest_signal['close'] > latest_htf_signal['avgma_htf']):
                    degis='dogru'
                else:
                    degis='yanlis'
                print(current_state, current_time, latest_htf_signal['avgma_htf'], latest_signal['close'], degis)
                time.sleep(5)  # sleep for 5 seconds to avoid performing the analysis multiple times in the buffer period
            else:
                time.sleep(1)  # sleep for 1 second
            retries = 0  # reset retries count after successful execution
        except ccxt.RequestTimeout:
            if retries < max_retries:
                retries += 1
                print(f"Request timed out, retrying ({retries}/{max_retries})...")
                time.sleep(1)  # wait for 1 second before retrying
            else:
                print("Maximum retries reached, stopping execution.")
                break  # exit the loop if maximum retries reached


if __name__ == '__main__':
    main()