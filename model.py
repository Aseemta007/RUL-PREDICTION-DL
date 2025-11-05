"""
🔧 Battery RUL Prediction — Optimized Model Training Script
-----------------------------------------------------------
This script:
1. Loads and analyzes training/test data.
2. Detects and fixes data quality issues automatically.
3. Builds and trains a robust LSTM model for Remaining Useful Life (RUL) prediction.
4. Saves the trained model and preprocessing artifacts for dashboard use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import pickle

# =============================================================================
# STEP 1 — Load Dataset
# =============================================================================
def load_data():
    """Load training and test datasets from the specified folders"""
    print("📂 Loading training and test data...")
    
    base_path = Path(r'C:\Users\aseem\OneDrive\Desktop\DEEP BATTERY\Dataset_Fixed')
    train_files = list((base_path / 'train').glob("*.csv"))
    test_files = list((base_path / 'test').glob("*.csv"))
    
    train_data = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    test_data = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
    
    print(f"✅ Training samples: {len(train_data)}, from {len(train_files)} files")
    print(f"✅ Test samples: {len(test_data)}, from {len(test_files)} files")
    return train_data, test_data


# =============================================================================
# STEP 2 — Analyze Data Quality
# =============================================================================
def analyze_data_quality(train_data, test_data):
    """Inspect RUL values, missing data, and feature correlations"""
    print("\n🔍 Analyzing data quality...")

    issues = []
    train_rul, test_rul = train_data['RUL_Cycles'], test_data['RUL_Cycles']

    # Check RUL ranges
    if train_rul.min() < 5 or test_rul.min() < 5:
        issues.append("LOW_RUL_VALUES")
    if train_rul.std() < 20:
        issues.append("LOW_RUL_VARIANCE")

    # Identify numeric features
    exclude_cols = ['Battery_ID', 'Date', 'Timestamp', 'RUL_Cycles', 'Charging_Strategy', 'Driving_Profile']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]

    # Compute correlation of each feature with RUL
    correlations = [(col, abs(np.corrcoef(train_data[col].fillna(0), train_rul)[0, 1]))
                    for col in feature_cols if not np.isnan(np.corrcoef(train_data[col].fillna(0), train_rul)[0, 1])]
    correlations.sort(key=lambda x: x[1], reverse=True)

    if correlations and correlations[0][1] < 0.3:
        issues.append("LOW_FEATURE_CORRELATION")

    if train_data.isnull().sum().sum() > 0 or test_data.isnull().sum().sum() > 0:
        issues.append("MISSING_VALUES")

    print(f"   → Top correlated feature: {correlations[0][0]} ({correlations[0][1]:.3f})")
    return issues, feature_cols, correlations


# =============================================================================
# STEP 3 — Fix Data Issues
# =============================================================================
def fix_data_issues(train_data, test_data, issues):
    """Apply fixes such as clipping, outlier removal, and NaN handling"""
    print("\n🩹 Fixing detected data issues...")

    train_data, test_data = train_data.copy(), test_data.copy()

    # Clip extremely small RUL values
    if "LOW_RUL_VALUES" in issues:
        train_data['RUL_Cycles'] = np.clip(train_data['RUL_Cycles'], 10, 2000)
        test_data['RUL_Cycles'] = np.clip(test_data['RUL_Cycles'], 10, 2000)

    # Remove RUL outliers (beyond ±3 std)
    mean, std = train_data['RUL_Cycles'].mean(), train_data['RUL_Cycles'].std()
    train_data = train_data[(train_data['RUL_Cycles'] >= mean - 3*std) & (train_data['RUL_Cycles'] <= mean + 3*std)]

    # Fill missing values with mean
    if "MISSING_VALUES" in issues:
        train_data.fillna(train_data.mean(), inplace=True)
        test_data.fillna(train_data.mean(), inplace=True)

    return train_data, test_data


# =============================================================================
# STEP 4 — Feature Selection
# =============================================================================
def select_best_features(correlations, top_k=20):
    """Select top K features based on correlation with RUL"""
    print(f"\n🎯 Selecting top {top_k} most predictive features...")
    selected = [feat for feat, _ in correlations[:top_k]]
    return selected


# =============================================================================
# STEP 5 — Create Sequential Data
# =============================================================================
def create_sequences(X, y, ids, seq_len=15, stride=3):
    """Convert tabular data into overlapping temporal sequences per battery"""
    sequences_X, sequences_y = [], []
    for battery in np.unique(ids):
        mask = ids == battery
        X_b, y_b = X[mask], y[mask]
        for i in range(0, len(X_b) - seq_len, stride):
            sequences_X.append(X_b[i:i+seq_len])
            sequences_y.append(y_b[i+seq_len])
    return np.array(sequences_X), np.array(sequences_y)


# =============================================================================
# STEP 6 — Build Optimized LSTM Model
# =============================================================================
def build_model(seq_len, n_features):
    """Define an optimized stacked LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2), BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2), BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.3), BatchNormalization(),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    return model


# =============================================================================
# STEP 7 — Train the Model
# =============================================================================
def train_model():
    """Main function: load data, process, train, and evaluate"""
    train_data, test_data = load_data()
    issues, feature_cols, correlations = analyze_data_quality(train_data, test_data)
    train_data, test_data = fix_data_issues(train_data, test_data, issues)
    selected_features = select_best_features(correlations, top_k=20)

    # Prepare feature matrices and labels
    X_train = train_data[selected_features].fillna(0).values
    y_train = train_data['RUL_Cycles'].values
    ids_train = train_data['Battery_ID'].values
    X_test = test_data[selected_features].fillna(0).values
    y_test = test_data['RUL_Cycles'].values
    ids_test = test_data['Battery_ID'].values

    # Create sequences
    SEQ_LEN, STRIDE = 15, 3
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, ids_train, SEQ_LEN, STRIDE)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, ids_test, SEQ_LEN, STRIDE)

    # Split validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_seq, y_train_seq, test_size=0.15, random_state=42)

    # Scale features and target
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    n_samp, n_steps, n_feat = X_train_final.shape
    X_train_final = scaler_X.fit_transform(X_train_final.reshape(-1, n_feat)).reshape(n_samp, n_steps, n_feat)
    X_val = scaler_X.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape[0], n_steps, n_feat)
    X_test_seq = scaler_X.transform(X_test_seq.reshape(-1, n_feat)).reshape(X_test_seq.shape[0], n_steps, n_feat)
    y_train_final = scaler_y.fit_transform(y_train_final.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_seq_scaled = scaler_y.transform(y_test_seq.reshape(-1, 1)).flatten()

    # Build and train model
    model = build_model(SEQ_LEN, n_feat)
    print("\n🚀 Training model...")
    output_dir = Path('outputs/models')
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(str(output_dir / 'best_model.keras'), monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=32, callbacks=callbacks, verbose=1
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test data...")
    y_pred_scaled = model.predict(X_test_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test_seq_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n✅ Test Results → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

    # Save model and scalers
    model.save(output_dir / 'fixed_model.keras')
    with open(output_dir / 'scalers.pkl', 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y, 'feature_names': selected_features, 'sequence_length': SEQ_LEN}, f)
    print("💾 Model and scalers saved successfully.")

    return model, history, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# =============================================================================
# RUN TRAINING
# =============================================================================
if __name__ == "__main__":
    print("="*80)
    print("⚙️  Starting Optimized Model Training for Battery RUL Prediction")
    print("="*80)
    model, history, metrics = train_model()
    print("\n🎉 Training Complete! Results:")
    print(metrics)
