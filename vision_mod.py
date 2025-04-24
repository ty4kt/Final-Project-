# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# | Script Section | >>> External Libraries <<< |
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# import sys
# import pandas as pd
# import hashlib
# from sklearn.preprocessing import StandardScaler, LabelEncoder



# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# | Script Section | >>> Calls <<< |
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# class Preprocessing:
#     def __init__(self, df):
#         self.df = df

#     def anonymize_column(self, column_name):
#         """Hashes the values in a specific column to anonymize them."""
#         self.df[column_name] = self.df[column_name].apply(
#             lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notnull(x) else x
#         )

#     def remove_sensitive_data(self):
#         """Removes or anonymizes sensitive columns."""
#         sensitive_columns = ['first', 'last', 'cc_num', 'gender', 'dob', 'job']
#         for col in sensitive_columns:
#             if col in self.df.columns:
#                 self.anonymize_column(col)
#         return self.df

    
#     def scale_numeric_data(self, exclusion_list=None):
#         """
#         Applies scaling to numeric columns in the dataset, excluding specified columns.

#         Parameters:
#             exclusion_list (list, optional): List of column names to exclude from scaling.
#                                              Defaults to ['is_fraud'].

#         Returns:
#             pd.DataFrame: The scaled dataset with specified columns left unchanged.
#         """
#         if exclusion_list is None:
#             exclusion_list = ['is_fraud']  # Default exclusion
        
#         # Identify numeric columns while excluding those in the exclusion list
#         numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
#         columns_to_scale = [col for col in numeric_columns if col not in exclusion_list]
        
#         # Apply scaling only to selected columns
#         scaler = StandardScaler()
#         self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
#         print(f"Scaling columns: {columns_to_scale}")

#         return self.df


#     def encode_categorical_data(self):
#         """Encodes all string-type columns to prepare them for machine learning."""
#         categorical_columns = self.df.select_dtypes(include=['object']).columns
#         label_encoders = {}
#         for col in categorical_columns:
#             label_encoders[col] = LabelEncoder()
#             self.df[col] = label_encoders[col].fit_transform(self.df[col].astype(str))
#         return self.df


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import sys
import pandas as pd
import hashlib
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessing:
    def __init__(self, df):
        self.df = df

    def anonymize_column(self, column_name):
        """Hashes the values in a specific column to anonymize them."""
        self.df[column_name] = self.df[column_name].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notnull(x) else x
        )

    def remove_sensitive_data(self):
        """Removes or anonymizes sensitive columns."""
        sensitive_columns = ['first', 'last', 'cc_num', 'gender', 'dob', 'job']
        for col in sensitive_columns:
            if col in self.df.columns:
                self.anonymize_column(col)
        return self.df

    def scale_numeric_data(self, exclusion_list=None):
        """
        Applies scaling to numeric columns in the dataset, excluding specified columns.
        """
        if exclusion_list is None:
            exclusion_list = ['is_fraud']
        
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        columns_to_scale = [col for col in numeric_columns if col not in exclusion_list]
        print(f"Scaling columns: {columns_to_scale}")  # Keep debug print
        
        scaler = StandardScaler()
        self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
        
        return self.df

    def encode_categorical_data(self):
        """Encodes all string-type columns to prepare them for machine learning, excluding 'is_fraud'."""
        categorical_columns = [col for col in self.df.select_dtypes(include=['object']).columns if col != 'is_fraud']
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            self.df[col] = label_encoders[col].fit_transform(self.df[col].astype(str))
        return self.df