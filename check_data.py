import pandas as pd
df = pd.read_csv('data/raw/kls_vdit_hourly_market.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('First timestamp:', df.iloc[0]['Timestamp'])
print('Last timestamp:', df.iloc[-1]['Timestamp'])
print('Unique years:', sorted(df['Timestamp'].str[:4].unique()))