import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt

ACC_COLS  = ['Var2', 'Var3', 'Var4']
GYRO_COLS = ['Var5', 'Var6', 'Var7']


def separate_gravity_body(df, fs=100, cutoff=20, acc_cols=None, gyro_cols=None):
    """
    Applies a Butterworth low-pass filter to separate gravitational and body
    components from accelerometer AND gyroscope signals.

    For accelerometers: gravity  = low-pass filtered signal
                        body     = raw - gravity
    For gyroscopes:     static   = low-pass filtered signal (slow drift/orientation)
                        dynamic  = raw - static            (fast rotational movement)
    """
    df = df.copy()

    if acc_cols is None:
        acc_cols  = [c for c in df.columns if 'acc' in c.lower()]
    if gyro_cols is None:
        gyro_cols = [c for c in df.columns if 'gyro' in c.lower() or 'gyr' in c.lower()]

    all_cols = acc_cols + gyro_cols
    if not all_cols:
        print("  Warning: No accelerometer or gyroscope columns found, skipping filter.")
        return df

    print(f"  Filtering acc cols:  {acc_cols}")
    print(f"  Filtering gyro cols: {gyro_cols}")

    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)

    # Map Var2/3/4 → x/y/z for readable output column names
    acc_axis_map  = {col: ax for col, ax in zip(acc_cols,  ['x', 'y', 'z'])}
    gyro_axis_map = {col: ax for col, ax in zip(gyro_cols, ['x', 'y', 'z'])}

    for col in acc_cols:
        ax = acc_axis_map[col]
        df[f"acc_{ax}_gravity"] = filtfilt(b, a, df[col].values)
        df[f"acc_{ax}_body"]    = df[col].values - df[f"acc_{ax}_gravity"].values

    for col in gyro_cols:
        ax = gyro_axis_map[col]
        df[f"gyro_{ax}_static"]  = filtfilt(b, a, df[col].values)
        df[f"gyro_{ax}_dynamic"] = df[col].values - df[f"gyro_{ax}_static"].values

    return df


def align_sensors(df_lower, df_upper, time_col='time', freq='10ms'):
    df_lower = df_lower.copy()
    df_upper = df_upper.copy()

    df_lower.index = pd.to_datetime(df_lower[time_col])
    df_upper.index = pd.to_datetime(df_upper[time_col])

    df_lower.drop(columns=[time_col], inplace=True)
    df_upper.drop(columns=[time_col], inplace=True)

    df_lower = df_lower[~df_lower.index.duplicated(keep='first')]
    df_upper = df_upper[~df_upper.index.duplicated(keep='first')]

    numeric_cols_lower = df_lower.select_dtypes(include='number').columns
    numeric_cols_upper = df_upper.select_dtypes(include='number').columns

    start = max(df_lower.index[0], df_upper.index[0])
    end   = min(df_lower.index[-1], df_upper.index[-1])
    common_grid = pd.date_range(start=start, end=end, freq=freq)

    def resample_to_grid(df, numeric_cols, grid):
        df_num = (df[numeric_cols]
                  .reindex(df.index.union(grid))
                  .interpolate(method='time')
                  .reindex(grid))
        non_numeric = [c for c in df.columns if c not in numeric_cols]
        if non_numeric:
            df_cat = (df[non_numeric]
                      .reindex(df.index.union(grid))
                      .ffill()
                      .bfill()
                      .reindex(grid))
            return pd.concat([df_num, df_cat], axis=1)
        return df_num

    df_lower = resample_to_grid(df_lower, numeric_cols_lower, common_grid)
    df_upper = resample_to_grid(df_upper, numeric_cols_upper, common_grid)

    return df_lower, df_upper


def sliding_window(data, window_size=128, step_size=64, label_col='Task', fs=100):
    """
    Extracts features from each sliding window:
      - Mean and std for all columns
      - Dominant FFT frequency for raw acc/gyro axes only (ACC_COLS + GYRO_COLS)
    """
    feature_cols = [c for c in data.columns if c != label_col]
    rows = []

    # Pre-compute frequency bins once — same for every window given fixed size
    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

    # Columns eligible for dominant frequency extraction
    fft_eligible_cols = ACC_COLS + GYRO_COLS

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        task = window[label_col].values[-1]
        window_features = window[feature_cols]

        mean_values = window_features.mean()
        std_values  = window_features.std()

        row_dict = {}

        for col in feature_cols:
            # ── Time-domain features ─────────────────────────────────────────
            row_dict[f"{col}_mean"] = mean_values[col]
            row_dict[f"{col}_std"]  = std_values[col]

            # ── Frequency-domain feature (raw axes only) ─────────────────────
            if col in fft_eligible_cols:
                signal = window[col].values
                fft_magnitude = np.abs(np.fft.rfft(signal))
                # Zero out DC component (index 0) to avoid gravity offset
                # dominating as a 0 Hz "frequency"
                fft_magnitude[0] = 0
                dominant_freq = freqs[np.argmax(fft_magnitude)]
                row_dict[f"{col}_dominant_freq"] = dominant_freq

        row_dict[label_col] = task
        rows.append(row_dict)

    return pd.DataFrame(rows)


def preprocess_lab_data(base_path="Raw_Data/Labeled", window_size=128, step_size=64, save_path="Processed_Data"):

    # ── Load all CSV files ───────────────────────────────────────────────────
    dataframes = []
    files = sorted(f for f in os.listdir(base_path) if f.endswith(".csv"))
    for filename in files:
        print(f"Loading {filename}...")
        df = pd.read_csv(os.path.join(base_path, filename), low_memory=False)
        dataframes.append(df)

    # ── Drop unused columns, fill missing labels ─────────────────────────────
    for df in dataframes:
        if 'Var1' in df.columns:
            df.drop(columns=['Var1'], inplace=True)
        if 'Task' in df.columns:
            df['Task'] = df['Task'].fillna('Unknown')

    # ── Align each lower/upper sensor pair ───────────────────────────────────
    aligned_pairs = []
    for i in range(0, len(dataframes), 2):
        print(f"Aligning sensor pair {i} (lower) and {i+1} (upper)...")
        df_lower, df_upper = align_sensors(
            dataframes[i], dataframes[i + 1], time_col='time', freq='10ms'
        )
        aligned_pairs.append((df_lower, df_upper))

    # ── Separate gravity/body and static/dynamic components ──────────────────
    filtered_pairs = []
    for pair_idx, (df_lower, df_upper) in enumerate(aligned_pairs):
        print(f"Separating gravity/body components for pair {pair_idx}...")
        df_lower = separate_gravity_body(df_lower, fs=100, cutoff=20,
                                         acc_cols=ACC_COLS, gyro_cols=GYRO_COLS)
        df_upper = separate_gravity_body(df_upper, fs=100, cutoff=20,
                                         acc_cols=ACC_COLS, gyro_cols=GYRO_COLS)
        filtered_pairs.append((df_lower, df_upper))

    # ── Sliding window + suffix renaming ─────────────────────────────────────
    session_dfs = []
    for pair_idx, (df_lower, df_upper) in enumerate(filtered_pairs):
        pair_windowed = []
        for sensor_df, suffix in [(df_lower, '_lower'), (df_upper, '_upper')]:
            print(f"  Applying sliding window to pair {pair_idx}, suffix {suffix}...")
            df_w = sliding_window(sensor_df, window_size=window_size,
                                  step_size=step_size, fs=100)
            new_columns = {col: f"{col}{suffix}" for col in df_w.columns if col != 'Task'}
            df_w.rename(columns=new_columns, inplace=True)
            print(f"  Shape: {df_w.shape}")
            pair_windowed.append(df_w)

        merged = pd.concat(pair_windowed, axis=1, join='inner')
        merged = merged.loc[:, ~merged.columns.duplicated()]
        session_dfs.append(merged)

    # ── Stack all sessions ────────────────────────────────────────────────────
    merged_df = pd.concat(session_dfs, axis=0).reset_index(drop=True)
    print(f"Final shape: {merged_df.shape}")

    if save_path:
        out_path = f"{save_path}/lab_W{window_size}_S{step_size}.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"Saved to '{out_path}'")

    return merged_df