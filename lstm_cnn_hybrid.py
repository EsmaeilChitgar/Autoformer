import pandas as pd

def csvEMA(ds_name='ETTh1.csv', remove_other_columns=False):
    # Load the CSV file
    df = pd.read_csv('./dataset/all_six_datasets/ETT-small/' + ds_name)

    # Define the span for the exponential moving average (EMA)
    span = 25  # Adjust this value as needed for smoothing

    # Calculate the exponential moving average
    df['EMA'] = df['OT'].ewm(span=span, adjust=False).mean()

    if remove_other_columns:
        # Keep only the 'date', 'LULL_EMA', and 'OT' columns
        df = df[['date', 'EMA', 'OT']]
    else:
        # Reorder columns to place 'LULL_EMA' before 'OT'
        columns = [col for col in df.columns if col != 'OT'] + ['OT']
        df = df[columns]

    # Save the modified DataFrame back to the original CSV file (overwrite)
    df.to_csv('./dataset/all_six_datasets/ETT-small/' + ds_name, index=False)

    # Display the first few rows of the modified DataFrame to verify
    print(df.head())

def add_3selected_features(ds_name='ETTh1.csv', save_name='ETTh1_selected_features.csv', remove_other_columns=True):
    # Load the dataset
    df = pd.read_csv('./dataset/all_six_datasets/ETT-small/' + ds_name)

    # Add LSTM lag feature (LSTM_Lag_5)
    df['LSTM_Lag_5'] = df['OT'].shift(5)

    # Add CNN moving average feature (CNN_MA_3)
    df['CNN_MA_3'] = df['OT'].rolling(window=3).mean()

    # Add Hybrid feature (Hybrid_MA_Lag10_5)
    df['Hybrid_MA_Lag10_5'] = df['OT'].shift(10).rolling(window=5).mean()

    # Drop rows with NaN values introduced by shifting/rolling
    df.dropna(inplace=True)

    if remove_other_columns:
        # Keep only the 'date', 'LULL_EMA', and 'OT' columns
        df = df[['date', 'LSTM_Lag_5', 'CNN_MA_3', 'Hybrid_MA_Lag10_5', 'OT']]
    else:
        # Reorder columns to place 'LULL_EMA' before 'OT'
        columns = [col for col in df.columns if col != 'OT'] + ['OT']
        df = df[columns]

    # Save the modified dataset with selected features
    df.to_csv('./dataset/all_six_datasets/ETT-small/' + save_name, index=False)

    # Display first few rows of the modified dataset to verify
    print(f"Selected feature dataset saved as: {save_name}")
    print(df[['LSTM_Lag_5', 'CNN_MA_3', 'Hybrid_MA_Lag10_5', 'OT']].head())

def add_lstm_features(df, seq_length=25):
    """
    Adds LSTM-style sequence features to the dataset.
    """
    for i in range(1, seq_length + 1):
        df[f'LSTM_Lag_{i}'] = df['OT'].shift(i)
    return df


def add_cnn_features(df, kernel_size=3):
    """
    Adds CNN-style sliding window average features to the dataset.
    """
    df[f'CNN_MA_{kernel_size}'] = df['OT'].rolling(window=kernel_size).mean()
    return df


def add_hybrid_features(df, seq_length=25, kernel_size=3):
    """
    Combines LSTM-style lag features and CNN-style moving averages.
    """
    df = add_lstm_features(df, seq_length)
    df = add_cnn_features(df, kernel_size)
    return df


def preprocess_with_features(ds_name='ETTm2.csv', save_name='ETTm2_processed.csv', seq_length=25, kernel_size=3):
    # Load the dataset
    df = pd.read_csv('./dataset/all_six_datasets/ETT-small/' + ds_name)

    # Add EMA as an auxiliary feature
    df['EMA'] = df['OT'].ewm(span=25, adjust=False).mean()

    # Add LSTM, CNN, and hybrid features
    df = add_hybrid_features(df, seq_length, kernel_size)

    # Drop rows with NaN values introduced by shifting/rolling
    df.dropna(inplace=True)

    # Save the augmented dataset
    df.to_csv('./dataset/all_six_datasets/ETT-small/' + save_name, index=False)
    print("Augmented dataset saved as:", save_name)
    print(df.head())
