

# import pandas as pd
# import os
# import numpy as np
# from vision_mod import Preprocessing
# from vision_helper import sample_df
# from vision_train import train_is_fraud_model

# # Define paths relative to the script's directory
# script_dir = os.path.dirname(os.path.abspath(__file__))
# payload_dir = os.path.join(script_dir, "..", "payload")
# csv_path = os.path.join(payload_dir, "fraudTrain.csv")
# save_path = payload_dir

# # Check if the CSV file exists
# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"Dataset not found at: {csv_path}")

# # Load the dataset
# print("Loading dataset...")
# df_payload = pd.read_csv(csv_path)

# # Ensure 'is_fraud' exists
# if 'is_fraud' not in df_payload.columns:
#     raise ValueError("Dataset must contain an 'is_fraud' column")

# # Print unique values and counts for debugging
# print("Unique values in 'is_fraud' before cleaning:", df_payload['is_fraud'].unique())
# print("Count of NaN in 'is_fraud':", df_payload['is_fraud'].isna().sum())
# print("Value counts in 'is_fraud':", df_payload['is_fraud'].value_counts(dropna=False).to_dict())

# # Map 'is_fraud' values: values near 0 or negative -> 0, values near 13 -> 1
# print("Mapping 'is_fraud' values...")
# df_payload['is_fraud'] = df_payload['is_fraud'].apply(
#     lambda x: 0 if x < 1 else 1
# ).astype(int)

# # Check for valid data after mapping
# if df_payload['is_fraud'].nunique() < 2:
#     raise ValueError("After mapping, 'is_fraud' contains only one class. Check the data.")

# # Print class distribution after mapping
# print(f"Class distribution after mapping: {df_payload['is_fraud'].value_counts().to_dict()}")

# # Debug trans_date_trans_time
# print("Sample trans_date_trans_time values:", df_payload['trans_date_trans_time'].head(10).tolist())
# print("Trans_date_trans_time NaN count:", df_payload['trans_date_trans_time'].isna().sum())

# # Add feature engineering (skip trans_hour due to invalid trans_date_trans_time)
# print("Adding engineered features...")
# df_payload['amt_log'] = np.log1p(df_payload['amt'])
# df_payload['city_pop_log'] = np.log1p(df_payload['city_pop'])

# # Handle NaN values in engineered features
# print("Handling NaN values in engineered features...")
# df_payload['amt_log'] = df_payload['amt_log'].fillna(df_payload['amt_log'].median())
# df_payload['city_pop_log'] = df_payload['city_pop_log'].fillna(df_payload['city_pop_log'].median())

# # Drop irrelevant and low-importance features
# print("Dropping irrelevant and low-importance features...")
# df_payload = df_payload.drop(columns=[
#     'Unnamed: 0', 'trans_num', 'street', 'merchant', 'unix_time', 'merch_lat', 'merch_long',
#     'cc_num', 'zip', 'long', 'trans_date_trans_time', 'gender', 'state', 'job', 'city_pop_log'
# ], errors='ignore')
# print("Columns after dropping:", df_payload.columns.tolist())

# # Check for NaNs in remaining features
# print("NaN counts in remaining features:", df_payload.isna().sum().to_dict())

# # Initialize the Preprocessing class
# print("Initializing preprocessing...")
# preprocessor = Preprocessing(df_payload)

# # Remove sensitive data
# print("Removing sensitive data...")
# processed_data = preprocessor.remove_sensitive_data()

# # Scale numeric data
# print("Scaling numeric data...")
# processed_data = preprocessor.scale_numeric_data()

# # Encode categorical data
# print("Encoding categorical data...")
# processed_data = preprocessor.encode_categorical_data()

# # Debug processed columns
# print("Columns after preprocessing:", processed_data.columns.tolist())

# # Check for NaNs after preprocessing
# print("NaN counts after preprocessing:", processed_data.isna().sum().to_dict())

# # Save the processed dataset
# output_file = os.path.join(save_path, 'df_out.csv')
# processed_data.to_csv(output_file, index=False)
# print(f"Sensitive data anonymized, numeric data scaled, and categorical data encoded. Saved as {output_file}")

# # Sample the processed data with stratification
# print("Sampling data...")
# sample_processed_data = sample_df(processed_data, n_samples=50000, random_state=42)
# sample_output_file = os.path.join(save_path, 'df_out_sample.csv')
# sample_processed_data.to_csv(sample_output_file, index=False)
# print(f"Sampled data saved as {sample_output_file}")

# # Print class distribution after sampling
# print(f"Class distribution in sampled data: {sample_processed_data['is_fraud'].value_counts().to_dict()}")

# # Ensure 'is_fraud' remains integer in sampled data
# sample_processed_data.loc[:, 'is_fraud'] = sample_processed_data['is_fraud'].astype(int)

# # Train the model on the sampled data
# print("Starting model training...")
# metrics = train_is_fraud_model(sample_processed_data)
# print("Model training completed. Performance metrics:")
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.4f}")

import pandas as pd
import os
import numpy as np
from vision_mod import Preprocessing
from vision_helper import sample_df
from vision_train import train_is_fraud_model

# Define paths relative to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
payload_dir = os.path.join(script_dir, "..", "payload")
csv_path = os.path.join(payload_dir, "fraudTrain.csv")
save_path = payload_dir

# Check if the CSV file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")

# Load the dataset
print("Loading dataset...")
df_payload = pd.read_csv(csv_path)

# Ensure 'is_fraud' exists
if 'is_fraud' not in df_payload.columns:
    raise ValueError("Dataset must contain an 'is_fraud' column")

# Print unique values and counts for debugging
print("Unique values in 'is_fraud' before cleaning:", df_payload['is_fraud'].unique())
print("Count of NaN in 'is_fraud':", df_payload['is_fraud'].isna().sum())
print("Value counts in 'is_fraud':", df_payload['is_fraud'].value_counts(dropna=False).to_dict())

# Map 'is_fraud' values: values near 0 or negative -> 0, values near 13 -> 1
print("Mapping 'is_fraud' values...")
df_payload['is_fraud'] = df_payload['is_fraud'].apply(
    lambda x: 0 if x < 1 else 1
).astype(int)

# Check for valid data after mapping
if df_payload['is_fraud'].nunique() < 2:
    raise ValueError("After mapping, 'is_fraud' contains only one class. Check the data.")

# Print class distribution after mapping
print(f"Class distribution after mapping: {df_payload['is_fraud'].value_counts().to_dict()}")

# Debug trans_date_trans_time
print("Sample trans_date_trans_time values:", df_payload['trans_date_trans_time'].head(10).tolist())
print("Trans_date_trans_time NaN count:", df_payload['trans_date_trans_time'].isna().sum())

# Add feature engineering (skip trans_hour due to invalid trans_date_trans_time)
print("Adding engineered features...")
df_payload['amt_log'] = np.log1p(df_payload['amt'])
df_payload['city_pop_log'] = np.log1p(df_payload['city_pop'])

# Handle NaN values in engineered features
print("Handling NaN values in engineered features...")
df_payload['amt_log'] = df_payload['amt_log'].fillna(df_payload['amt_log'].median())
df_payload['city_pop_log'] = df_payload['city_pop_log'].fillna(df_payload['city_pop_log'].median())

# Drop irrelevant features
print("Dropping irrelevant features...")
df_payload = df_payload.drop(columns=[
    'Unnamed: 0', 'trans_num', 'street', 'merchant', 'unix_time', 'merch_lat', 'merch_long',
    'cc_num', 'zip', 'long', 'trans_date_trans_time'
], errors='ignore')
print("Columns after dropping:", df_payload.columns.tolist())

# Check for NaNs in remaining features
print("NaN counts in remaining features:", df_payload.isna().sum().to_dict())

# Initialize the Preprocessing class
print("Initializing preprocessing...")
preprocessor = Preprocessing(df_payload)

# Remove sensitive data
print("Removing sensitive data...")
processed_data = preprocessor.remove_sensitive_data()

# Scale numeric data
print("Scaling numeric data...")
processed_data = preprocessor.scale_numeric_data()

# Encode categorical data
print("Encoding categorical data...")
processed_data = preprocessor.encode_categorical_data()

# Debug processed columns
print("Columns after preprocessing:", processed_data.columns.tolist())

# Check for NaNs after preprocessing
print("NaN counts after preprocessing:", processed_data.isna().sum().to_dict())

# Save the processed dataset
output_file = os.path.join(save_path, 'df_out.csv')
processed_data.to_csv(output_file, index=False)
print(f"Sensitive data anonymized, numeric data scaled, and categorical data encoded. Saved as {output_file}")

# Sample the processed data with stratification
print("Sampling data...")
sample_processed_data = sample_df(processed_data, n_samples=50000, random_state=42)
sample_output_file = os.path.join(save_path, 'df_out_sample.csv')
sample_processed_data.to_csv(sample_output_file, index=False)
print(f"Sampled data saved as {sample_output_file}")

# Print class distribution after sampling
print(f"Class distribution in sampled data: {sample_processed_data['is_fraud'].value_counts().to_dict()}")

# Ensure 'is_fraud' remains integer in sampled data
sample_processed_data.loc[:, 'is_fraud'] = sample_processed_data['is_fraud'].astype(int)

# Train the model on the sampled data
print("Starting model training...")
metrics = train_is_fraud_model(sample_processed_data)
print("Model training completed. Performance metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")