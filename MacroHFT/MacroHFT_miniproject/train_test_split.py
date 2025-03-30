import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Mount Google Drive

# Set base path to your dataset folder in Google Drive
base_path = 'C:/Users/KIIT/OneDrive/Desktop/AI'  # Update the path if your folder name is different

etfs_path = os.path.join(base_path, 'etfs')
stocks_path = os.path.join(base_path, 'stocks')
symbols_meta_path = os.path.join(base_path, 'symbols_valid_meta.xls')

# Load metadata (if required)
try:
    symbols_meta = pd.read_excel(symbols_meta_path)
    print("✅ Metadata loaded successfully!")
    print(f"Metadata shape: {symbols_meta.shape}")
except FileNotFoundError:
    print("❌ Metadata file not found. Proceeding without it.")
    symbols_meta = None
except Exception as e:
    print(f"⚠️ An error occurred while loading metadata: {e}")
    symbols_meta = None

# Load and combine ETF CSV files
etf_files = [os.path.join(etfs_path, file) for file in os.listdir(etfs_path) if file.endswith('.csv')]
etf_data = pd.concat([pd.read_csv(file, low_memory=False) for file in etf_files], ignore_index=True)
print(f"✅ Loaded {len(etf_files)} ETF files with shape: {etf_data.shape}")

# Load and combine Stock CSV files
stock_files = [os.path.join(stocks_path, file) for file in os.listdir(stocks_path) if file.endswith('.csv')]
stock_data = pd.concat([pd.read_csv(file, low_memory=False) for file in stock_files], ignore_index=True)
print(f"✅ Loaded {len(stock_files)} Stock files with shape: {stock_data.shape}")

# Merge ETF, Stock data, and optionally symbols_meta if loaded
all_data = pd.concat([etf_data, stock_data], ignore_index=True)

if symbols_meta is not None:
    all_data = all_data.merge(symbols_meta, how='left', left_on='symbol', right_on='symbol')  # Adjust 'symbol' column if necessary

if symbols_meta is not None:
    print("📄 Company Names in Metadata:")
    print(symbols_meta['company'].unique())
print(f"🔹 Combined data shape: {all_data.shape}")

# Limit the data size if memory is insufficient
if len(all_data) > 5_000_000:  # Adjust this limit based on available memory in Colab
    all_data = all_data.sample(2_000_000, random_state=42)

# Split into train, test, validate
df_train, df_temp = train_test_split(all_data, test_size=0.3, random_state=42)
df_test, df_validate = train_test_split(df_temp, test_size=0.33, random_state=42)

# Save CSV files back to Google Drive
output_path = os.path.join(base_path, 'output_offline')
os.makedirs(output_path, exist_ok=True)

df_train.to_csv(os.path.join(output_path, 'df_train.csv'), index=False)
df_test.to_csv(os.path.join(output_path, 'df_test.csv'), index=False)
df_validate.to_csv(os.path.join(output_path, 'df_validate.csv'), index=False)

print("✅ Train, Test, and Validate CSV files created successfully!")
