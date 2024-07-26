import lib.db as db
import lib.redis as redis
import lib.util as util
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lib.data_preprocessing import parse_orderbook, features
from tkan import TKAN

exchange = 'vertex-perp'
pair = 'BTC-USDC-SWAP'
fit_time_range = util.get_prev_hour_ms()
predict_time_range = 60

if __name__ == '__main__':
    print('----------------- Data Preprocessing')

    if not util.find_files_by_name('chunk', exchange):
        print('----------------- Load from db and save to file')
        db.get_query_to_file('SELECT timestamp, asks, bids, spread, mid_price FROM orderbook_new WHERE exchange = %s AND pair = %s AND timestamp > %s', exchange, (exchange, pair, fit_time_range))

    files = util.find_files_by_name('chunk', exchange)
    processed_dfs = []
    for file in files:
        df = pd.read_csv(file)
        df[features] = df.apply(parse_orderbook, axis=1)
        df.drop(columns=['asks', 'bids'], inplace=True)
        processed_dfs.append(df)
        print(f'----------------- {file} done')

    orderbook_df = pd.concat(processed_dfs, ignore_index=True)
    orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'], unit='ms')
    orderbook_df.set_index('timestamp', inplace=True)
    orderbook_df.sort_values(by='timestamp', inplace=True, ascending=True)

    print('----------------- Data Resampling')

    orderbook_df_resampled = orderbook_df.resample('100ms').mean()
    orderbook_df_resampled = orderbook_df_resampled.interpolate(method='linear')

    X = []
    y = []

    print(orderbook_df_resampled)

    for i in range(len(orderbook_df_resampled) - predict_time_range):
        X.append(orderbook_df_resampled[features].iloc[i:i + predict_time_range].values)
        y.append(orderbook_df_resampled['mid_price'].iloc[i + predict_time_range])

    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)

    print('----------------- Model Fit')

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(predict_time_range, len(features))),
        TKAN(100, tkan_activations=[
            {'spline_order': 3, 'grid_size': 10},
            {'spline_order': 1, 'grid_size': 5},
            {'spline_order': 4, 'grid_size': 6}
        ], return_sequences=True, use_bias=True),
        TKAN(100, tkan_activations=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True),
        TKAN(100, tkan_activations=['relu', 'relu', 'relu', 'relu', 'relu'], return_sequences=True, use_bias=True),
        TKAN(100, tkan_activations=[None for _ in range(3)], return_sequences=False, use_bias=True),
        tf.keras.layers.Dense(1),
    ])

    model.compile(optimizer='adam', loss='mse')

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 모델 학습
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    model.fit(X_train, y_train, epochs=1500, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
    model.save('model.h5')
