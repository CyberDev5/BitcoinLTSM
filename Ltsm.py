import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime

# Configuration de l'API Binance
api_key = 'ma clé publique binance'
api_secret = 'ma clé privée binance'
client = Client(api_key, api_secret)

def get_all_binance(symbol, interval, start_str, end_str=None):
    """ Get all binance historical data for a symbol """
    timeframe = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200
    }
    limit = 1000  # Binance API limit
    timeframe_ms = timeframe[interval] * 60 * 1000
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else int(datetime.now().timestamp() * 1000)
    all_klines = []
    
    while start_ts < end_ts:
        temp_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=min(start_ts + limit * timeframe_ms, end_ts),
            limit=limit
        )
        if not temp_klines:
            break
        all_klines += temp_klines
        start_ts = temp_klines[-1][0] + timeframe_ms
    
    return all_klines

# Exemple d'utilisation
data = get_all_binance('BTCUSDT', '15m', '2023-01-01', '2023-12-31')
df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
def get_ticksize(data, freq=30):
    numlen = int(len(data)/2)
    tztail = data.tail(numlen).copy()
    tztail['tz'] = tztail.Close.rolling(freq).std()  
    tztail = tztail.dropna()
    ticksize = np.ceil(tztail['tz'].mean()*0.25)  

    if ticksize < 0.2:
        ticksize = 0.2  

    return int(ticksize)

def abc(session_hr=6.5, freq=30):
    caps = [chr(i) for i in range(65, 91)]
    abc_lw = [x.lower() for x in caps]
    Aa = caps + abc_lw
    alimit = math.ceil(session_hr * (60 / freq)) + 3
    if alimit > 52:
        alphabets = Aa * int((np.ceil((alimit - 52) / 52)) + 1)
    else:
        alphabets = Aa[0:alimit]
    return alphabets

def tpo(dft_rs, freq=30, ticksize=10, style='tpo', session_hr=6.5):
    if len(dft_rs) > int(60/freq):
        dft_rs = dft_rs.drop_duplicates('timestamp')
        dft_rs = dft_rs.reset_index(inplace=False, drop=True)
        dft_rs['rol_mx'] = dft_rs['High'].cummax()
        dft_rs['rol_mn'] = dft_rs['Low'].cummin()
        dft_rs['ext_up'] = dft_rs['rol_mn'] > dft_rs['rol_mx'].shift(2)
        dft_rs['ext_dn'] = dft_rs['rol_mx'] < dft_rs['rol_mn'].shift(2)

        alphabets = abc(session_hr, freq)
        alphabets = alphabets[0:len(dft_rs)]
        hh = dft_rs['High'].max()
        ll = dft_rs['Low'].min()
        day_range = hh - ll
        dft_rs['abc'] = alphabets
        place = int(np.ceil((hh - ll) / ticksize))
        abl_bg = []
        tpo_countbg = []
        pricel = []
        volcountbg = []

        for u in range(place):
            abl = []
            tpoc = []
            volcount = []
            p = ll + (u*ticksize)
            for lenrs in range(len(dft_rs)):
                if p >= dft_rs['Low'][lenrs] and p < dft_rs['High'][lenrs]:
                    abl.append(dft_rs['abc'][lenrs])
                    tpoc.append(1)
                    volcount.append((dft_rs['Volume'][lenrs]) / freq)
            abl_bg.append(''.join(abl))
            tpo_countbg.append(sum(tpoc))
            volcountbg.append(sum(volcount))
            pricel.append(p)

        dftpo = pd.DataFrame({'close': pricel, 'alphabets': abl_bg, 'tpocount': tpo_countbg, 'volsum': volcountbg})
        dftpo['alphabets'].replace('', np.nan, inplace=True)
        dftpo = dftpo.dropna()
        dftpo = dftpo.reset_index(inplace=False, drop=True)
        dftpo = dftpo.sort_index(ascending=False)
        dftpo = dftpo.reset_index(inplace=False, drop=True)

        if style == 'tpo':
            column = 'tpocount'
        else:
            column = 'volsum'

        dfmx = dftpo[dftpo[column] == dftpo[column].max()]

        mid = ll + ((hh - ll) / 2)
        dfmax = dfmx.copy()
        dfmax['poc-mid'] = abs(dfmax['close'] - mid)
        pocidx = dfmax['poc-mid'].idxmin()
        poc = dfmax['close'][pocidx]
        poctpo = dftpo[column].max()
        tpo_updf = dftpo[dftpo['close'] > poc]
        tpo_updf = tpo_updf.sort_index(ascending=False)
        tpo_updf = tpo_updf.reset_index(inplace=False, drop=True)

        tpo_dndf = dftpo[dftpo['close'] < poc]
        tpo_dndf = tpo_dndf.reset_index(inplace=False, drop=True)

        valtpo = (dftpo[column].sum()) * 0.70

        abovepoc = tpo_updf[column].to_list()
        belowpoc = tpo_dndf[column].to_list()

        if (len(abovepoc)/2).is_integer() is False:
            abovepoc = abovepoc+[0]

        if (len(belowpoc)/2).is_integer() is False:
            belowpoc = belowpoc+[0]

        bel2 = np.array(belowpoc).reshape(-1, 2)
        bel3 = bel2.sum(axis=1)
        bel4 = list(bel3)
        abv2 = np.array(abovepoc).reshape(-1, 2)
        abv3 = abv2.sum(axis=1)
        abv4 = list(abv3)
        df_va = pd.DataFrame({'abv': pd.Series(abv4), 'bel': pd.Series(bel4)})
        df_va = df_va.fillna(0)
        df_va['abv_idx'] = np.where(df_va.abv > df_va.bel, 1, 0)
        df_va['bel_idx'] = np.where(df_va.bel > df_va.abv, 1, 0)
        df_va['cum_tpo'] = np.where(df_va.abv > df_va.bel, df_va.abv, 0)
        df_va['cum_tpo'] = np.where(df_va.bel > df_va.abv, df_va.bel, df_va.cum_tpo)

        df_va['cum_tpo'] = np.where(df_va.abv == df_va.bel, df_va.abv+df_va.bel, df_va.cum_tpo)
        df_va['abv_idx'] = np.where(df_va.abv == df_va.bel, 1, df_va.abv_idx)
        df_va['bel_idx'] = np.where(df_va.abv == df_va.bel, 1, df_va.bel_idx)
        df_va['cum_tpo_cumsum'] = df_va.cum_tpo.cumsum()
        df_va_cut = df_va[df_va.cum_tpo_cumsum + poctpo <= valtpo]
        vah_idx = (df_va_cut.abv_idx.sum())*2
        val_idx = (df_va_cut.bel_idx.sum())*2

        if vah_idx >= len(tpo_updf) and vah_idx != 0:
            vah_idx = vah_idx - 2

        if val_idx >= len(tpo_dndf) and val_idx != 0:
            val_idx = val_idx - 2

        vah = tpo_updf.close[vah_idx]
        val = tpo_dndf.close[val_idx]

        tpoval = dftpo[ticksize * 2:-(ticksize * 2)]['tpocount']
        exhandle_index = np.where(tpoval <= 2, tpoval.index, None)
        exhandle_index = list(filter(None, exhandle_index))
        distance = ticksize * 3
        lvn_list = []
        for ex in exhandle_index[0:-1:distance]:
            lvn_list.append(dftpo['close'][ex])

        excess_h = dftpo[0:ticksize]['tpocount'].sum() / ticksize
        excess_l = dftpo[-(ticksize):]['tpocount'].sum() / ticksize
        excess = 0
        if excess_h == 1 and dftpo.iloc[-1]['close'] < poc:
            excess = dftpo['close'][ticksize]

        if excess_l == 1 and dftpo.iloc[-1]['close'] >= poc:
            excess = dftpo.iloc[-ticksize]['close']

        area_above_poc = dft_rs.High.max() - poc
        area_below_poc = poc - dft_rs.Low.min()
        if area_above_poc == 0:
            area_above_poc = 1
        if area_below_poc == 0:
            area_below_poc = 1
        balance = area_above_poc/area_below_poc

        if balance >= 0:
            bal_target = poc - area_above_poc
        else:
            bal_target = poc + area_below_poc

        mp = {'df': dftpo, 'vah': round(vah, 2), 'poc': round(poc, 2), 'val': round(val, 2), 'lvn': lvn_list, 'excess': round(excess, 2),
              'bal_target': round(bal_target, 2)}

    else:
        print('not enough bars for date {}'.format(dft_rs['timestamp'][0]))
        mp = {}

    return mp

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Préparation des données pour le LSTM
def prepare_data(df, lookback=50):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Close', 'Volume', 'High', 'Low']])
    X, y = [], []
    for i in range(len(scaled_data) - lookback - 1):
        X.append(scaled_data[i:(i + lookback)])
        y.append(scaled_data[i + lookback, 0])  # Prédire la fermeture
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Création du modèle LSTM
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Exemple d'utilisation
lookback = 50
X, y, scaler = prepare_data(df)
model = create_lstm_model((X.shape[1], X.shape[2]))
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

from tensorflow.keras.layers import Attention, Concatenate

# Ajout du mécanisme d'attention
def create_lstm_attention_model(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    lstm_out = LSTM(128, return_sequences=True)(input_layer)
    attention = Attention()([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attention])
    lstm_out_2 = LSTM(64, return_sequences=False)(concat)
    output = Dense(1)(lstm_out_2)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Exemple d'utilisation
attention_model = create_lstm_attention_model((X.shape[1], X.shape[2]))
attention_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Visualisation du profil de marché en formation
def plot_market_profile(mp_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(mp_data['close'], mp_data['tpocount'], color='blue', edgecolor='black')
    ax.axvline(x=mp_data['vah'], color='green', linestyle='--', label='VAH')
    ax.axvline(x=mp_data['val'], color='red', linestyle='--', label='VAL')
    ax.axvline(x=mp_data['poc'], color='orange', linestyle='--', label='POC')
    ax.legend()
    plt.xlabel('TPO Count')
    plt.ylabel('Price')
    plt.title('Market Profile')
    plt.show()

# Exemple d'utilisation
mp_data = tpo(df)
plot_market_profile(mp_data)

import smtplib
from email.mime.text import MIMEText

def send_alert(pattern, probability, direction):
    msg = MIMEText(f"Pattern: {pattern}\nProbability: {probability}%\nDirection: {direction}")
    msg['Subject'] = 'Market Profile Alert'
    msg['From'] = 'email@gmail.com'
    msg['To'] = 'email@gmail.com'

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.login('@gmail.com', 'your_password')
        server.sendmail('@gmail.com', '@gmail.com', msg.as_string())

# Exemple d'utilisation
send_alert('Open-drive', 83, 'Bullish')
