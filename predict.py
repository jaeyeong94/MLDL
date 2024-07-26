import numpy as np
import tensorflow as tf
from tkan import TKAN

# 데이터 생성
X_train_seq = np.random.rand(100, 10, 20)
y_train_seq = np.random.rand(100, 1)

# TKAN 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=X_train_seq.shape[1:]),
    TKAN(100, tkan_activations=[
        {'spline_order': 3, 'grid_size': 10},
        {'spline_order': 1, 'grid_size': 5},
        {'spline_order': 4, 'grid_size': 6}
    ], return_sequences=True, use_bias=True),
    TKAN(100, tkan_activations=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True),
    TKAN(100, tkan_activations=['relu', 'relu', 'relu', 'relu', 'relu'], return_sequences=True, use_bias=True),
    TKAN(100, tkan_activations=[None for _ in range(3)], return_sequences=False, use_bias=True),
    tf.keras.layers.Dense(y_train_seq.shape[1]),
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=16)

# 예측 수행
X_new = np.random.rand(10, 10, 20)  # 새로운 입력 데이터
y_pred = model.predict(X_new)

print(y_pred)
