# WTC-HNet
# Advanced WT-TVE-Hybrid Model with Bold Visualizations + Hybrid Scaling (Pollution Target)
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input

# Load dataset
file_path = '/content/LSTM-Multivariate_pollution.csv'
df = pd.read_csv(file_path, index_col='date', parse_dates=True)
if df['pollution'].dtype == object:
    df['pollution'] = pd.to_numeric(df['pollution'].str.replace(',', ''), errors='coerce')
else:
    df['pollution'] = pd.to_numeric(df['pollution'], errors='coerce')
prices = df['pollution'].values.reshape(-1, 1)

# Hybrid Scaling: (X - mean) / (max - min)
def hybrid_scale(data):
    mean = np.mean(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - mean) / (max_val - min_val + 1e-8)  # Avoid division by zero

prices_scaled = hybrid_scale(prices)

# Wavelet Transform
def apply_wavelet_transform(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data.flatten(), wavelet, level=level)
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed[:len(data)].reshape(-1, 1), coeffs

prices_wavelet, wavelet_coeffs = apply_wavelet_transform(prices_scaled)

# Bold Visualization: Original vs Reconstructed
plt.figure(figsize=(12, 6))
plt.plot(prices_scaled, label='Original', linewidth=2.5)
plt.plot(prices_wavelet, label='Wavelet Reconstructed', linestyle='--', linewidth=2.5)
plt.title('Original vs Wavelet Reconstructed Signal', fontweight='bold', fontsize=16)
plt.xlabel('Time', fontweight='bold')
plt.ylabel('Normalized Price', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linewidth=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.show()

# Wavelet Coefficients
plt.figure(figsize=(12, 6))
for i, coeff in enumerate(wavelet_coeffs[1:], 1):
    plt.subplot(len(wavelet_coeffs) - 1, 1, i)
    plt.plot(coeff, label=f'Detail Coefficients Level {i}', linewidth=2)
    plt.legend()
plt.suptitle('Wavelet Detail Coefficients', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# Temporal Variance Embedding (without SVD)
def temporal_variance_embedding(data, window=10):
    return np.array([np.var(data[i-window:i]) for i in range(window, len(data))]).reshape(-1, 1)

variance_features = temporal_variance_embedding(prices_wavelet, window=10)

# Sequence creation
SEQ_LEN = 30
def create_sequences(data, variance, seq_length=SEQ_LEN):
    X, y = [], []
    for i in range(len(variance) - seq_length + 1):
        X.append(np.hstack((data[i:i+seq_length], variance[i:i+seq_length])))
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(prices_wavelet, variance_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Functional API Hybrid Model
def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs=inputs, outputs=outputs)

model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_test, y_test),
                    callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')

# Bold Results Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', color='blue', linewidth=2.5)
plt.plot(y_pred, label='Predicted', color='red', linewidth=2.5)
plt.title('Air Pollution Prediction using WT-TVE-HybridNet', fontweight='bold', fontsize=16)
plt.xlabel('Time', fontweight='bold')
plt.ylabel('Pollution', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linewidth=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.show()

# Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2.5)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5)
plt.title('Model Loss over Epochs', fontweight='bold', fontsize=16)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linewidth=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.show()
# Advanced WT-TVE-Hybrid-CA Model with Bold Visualizations + Hybrid Scaling (Pollution Target)
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input

# Load dataset
file_path = '/content/LSTM-Multivariate_pollution.csv'
df = pd.read_csv(file_path, index_col='date', parse_dates=True)
if df['pollution'].dtype == object:
    df['pollution'] = pd.to_numeric(df['pollution'].str.replace(',', ''), errors='coerce')
else:
    df['pollution'] = pd.to_numeric(df['pollution'], errors='coerce')
prices = df['pollution'].values.reshape(-1, 1)

# Hybrid Scaling: (X - mean) / (max - min)
def hybrid_scale(data):
    mean = np.mean(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - mean) / (max_val - min_val + 1e-8)  # Avoid division by zero

prices_scaled = hybrid_scale(prices)

# Wavelet Transform
def apply_wavelet_transform(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data.flatten(), wavelet, level=level)
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed[:len(data)].reshape(-1, 1), coeffs

prices_wavelet, wavelet_coeffs = apply_wavelet_transform(prices_scaled)

# --------- Cellular Automata Block -----------
def apply_cellular_automata(data, steps=3):
    ca_data = data.copy()
    for _ in range(steps):
        left_shift = np.roll(ca_data, 1)
        right_shift = np.roll(ca_data, -1)
        ca_data = (left_shift + ca_data + right_shift) / 3  # Simple averaging rule
    return ca_data

prices_ca = apply_cellular_automata(prices_wavelet)

# Bold Visualization: Original vs CA Enhanced
plt.figure(figsize=(12, 6))
plt.plot(prices_wavelet, label='Wavelet', linewidth=2.5)
plt.plot(prices_ca, label='CA Enhanced', linestyle='--', linewidth=2.5)
plt.title('Wavelet vs Cellular Automata Enhanced Signal', fontweight='bold', fontsize=16)
plt.xlabel('Time', fontweight='bold')
plt.ylabel('Normalized Price', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linewidth=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.show()

# Wavelet Coefficients
plt.figure(figsize=(12, 6))
for i, coeff in enumerate(wavelet_coeffs[1:], 1):
    plt.subplot(len(wavelet_coeffs) - 1, 1, i)
    plt.plot(coeff, label=f'Detail Coefficients Level {i}', linewidth=2)
    plt.legend()
plt.suptitle('Wavelet Detail Coefficients', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# Temporal Variance Embedding (without SVD)
def temporal_variance_embedding(data, window=10):
    return np.array([np.var(data[i-window:i]) for i in range(window, len(data))]).reshape(-1, 1)

variance_features = temporal_variance_embedding(prices_ca, window=10)

# Sequence creation
SEQ_LEN = 30
def create_sequences(data, variance, seq_length=SEQ_LEN):
    X, y = [], []
    for i in range(len(variance) - seq_length + 1):
        X.append(np.hstack((data[i:i+seq_length], variance[i:i+seq_length])))
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(prices_ca, variance_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Functional API Hybrid Model
def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs=inputs, outputs=outputs)

model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_test, y_test),
                    callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')

# Bold Results Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', color='blue', linewidth=2.5)
plt.plot(y_pred, label='Predicted', color='red', linewidth=2.5)
plt.title('Air Pollution Prediction using WT-TVE-CA-HybridNet', fontweight='bold', fontsize=16)
plt.xlabel('Time', fontweight='bold')
plt.ylabel('Pollution', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linewidth=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.show()

# Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2.5)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5)
plt.title('Model Loss over Epochs', fontweight='bold', fontsize=16)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linewidth=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.show()
