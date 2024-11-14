import yfinance as yf


def fetch_bitcoin_prices():
    # Fetch historical data for Bitcoin from Yahoo Finance
    btc_data = yf.download('BTC-USD', start='2021-01-01', end='2024-11-14', interval='1d')

    # Rename columns to match your requirements
    btc_data.reset_index(inplace=True)
    btc_data.rename(columns={
        'Date': 'Date',
        'Open': 'Open Price',
        'High': 'High Price',
        'Low': 'Low Price',
        'Close': 'Close Price',
        'Adj Close': 'Adjusted Close Price',
        'Volume': 'Volume'
    }, inplace=True)

    # Save as CSV
    btc_data.to_csv('dataset/all_six_datasets/ETT-small/bitcoin_prices4.csv', index=False)
    print("Bitcoin prices saved to bitcoin_prices_last_3_years.csv")


if __name__ == '__main__':
    fetch_bitcoin_prices()
