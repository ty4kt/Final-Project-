

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# def train_is_fraud_model(data: pd.DataFrame):
#     """
#     Trains an ensemble model (Random Forest + XGBoost) to predict fraud with SMOTE, cross-validation, and threshold optimization.

#     Parameters:
#         data (pd.DataFrame): The dataset containing features and a binary target column named 'is_fraud'.

#     Returns:
#         dict: A dictionary containing model performance metrics.
#     """
#     try:
#         if 'is_fraud' not in data.columns:
#             raise ValueError("Dataset must contain an 'is_fraud' column as the target variable.")

#         # Validate labels
#         unique_labels = data['is_fraud'].unique()
#         if not set(unique_labels).issubset({0, 1}):
#             raise ValueError(f"'is_fraud' contains invalid labels: {unique_labels}. Expected [0, 1].")

#         # Drop known low-importance features
#         drop_cols = ['gender', 'first', 'last', 'dob', 'job', 'state', 'city', 'city_pop']
#         data = data.drop(columns=drop_cols, errors='ignore')

#         # Feature engineering
#         if 'amt' in data.columns and 'city_pop' in data.columns:
#             data['amt_per_pop'] = data['amt'] / (data['city_pop'] + 1e-5)

#         # Split into features and target
#         print("Splitting data into features and target...")
#         X = data.drop(columns=['is_fraud'])
#         y = data['is_fraud']

#         # Check for NaNs
#         if X.isna().any().any():
#             print("NaNs found in features:", X.isna().sum().to_dict())
#             raise ValueError("Input features contain NaNs. Please clean the data.")

#         # Split into train/test sets
#         print("Splitting data into train and test sets...")
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#         print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")

#         # Apply SMOTE for balancing
#         print("Applying SMOTE...")
#         smote = SMOTE(sampling_strategy=0.25, random_state=42)
#         X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#         print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

#         # Standardize
#         print("Standardizing features...")
#         scaler = StandardScaler()
#         X_train_res = scaler.fit_transform(X_train_res)
#         X_test_scaled = scaler.transform(X_test)

#         # Initialize individual models
#         rf = RandomForestClassifier(
#             n_estimators=300,
#             max_depth=25,
#             class_weight='balanced_subsample',
#             random_state=42
#         )

#         xgb = XGBClassifier(
#             n_estimators=300,
#             max_depth=10,
#             scale_pos_weight=5,
#             use_label_encoder=False,
#             eval_metric='logloss',
#             random_state=42
#         )

#         # Ensemble model
#         print("Initializing ensemble model...")
#         ensemble = VotingClassifier(
#             estimators=[('rf', rf), ('xgb', xgb)],
#             voting='soft',
#             weights=[1, 2]  # Give more weight to XGBoost
#         )

#         print("Performing cross-validation...")
#         cv_scores = cross_val_score(ensemble, X_train_res, y_train_res, cv=5, scoring='f1')
#         print(f"Cross-validation F1 scores: {cv_scores}")
#         print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

#         print("Training ensemble model...")
#         ensemble.fit(X_train_res, y_train_res)

#         print("Testing multiple thresholds...")
#         thresholds = [round(t, 2) for t in np.arange(0.2, 0.81, 0.05)]
#         best_fbeta = 0
#         best_threshold = 0.2
#         best_metrics = {}
#         y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

#         for threshold in thresholds:
#             print(f"Evaluating threshold: {threshold}")
#             y_pred = (y_pred_proba >= threshold).astype(int)
#             metrics = {
#                 "accuracy": accuracy_score(y_test, y_pred),
#                 "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
#                 "recall": recall_score(y_test, y_pred, pos_label=1),
#                 "f1_score": f1_score(y_test, y_pred, pos_label=1),
#                 "fbeta_score": fbeta_score(y_test, y_pred, beta=2, pos_label=1)
#             }
#             print(f"Metrics at threshold {threshold}:")
#             for metric, value in metrics.items():
#                 print(f"{metric}: {value:.4f}")
#             print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#             if metrics['fbeta_score'] > best_fbeta:
#                 best_fbeta = metrics['fbeta_score']
#                 best_threshold = threshold
#                 best_metrics = metrics

#         print(f"Best threshold: {best_threshold} (Fβ-score: {best_fbeta:.4f})")
#         print("Final Confusion Matrix:\n", confusion_matrix(y_test, (y_pred_proba >= best_threshold).astype(int)))

#         return best_metrics

#     except Exception as e:
#         print(f"Error in train_is_fraud_model: {e}")
#         raise


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


# def train_is_fraud_model(data: pd.DataFrame):
#     """
#     Trains a stacked ensemble model (Random Forest, XGBoost, LightGBM + Logistic Regression meta-model)
#     with SMOTE, cross-validation, feature engineering, and threshold optimization.

#     Parameters:
#         data (pd.DataFrame): The dataset containing features and a binary target column named 'is_fraud'.

#     Returns:
#         dict: A dictionary containing model performance metrics.
#     """
#     try:
#         if 'is_fraud' not in data.columns:
#             raise ValueError("Dataset must contain an 'is_fraud' column as the target variable.")

#         # Validate labels
#         unique_labels = data['is_fraud'].unique()
#         if not set(unique_labels).issubset({0, 1}):
#             raise ValueError(f"'is_fraud' contains invalid labels: {unique_labels}. Expected [0, 1].")

#         # Drop known low-importance features
#         drop_cols = ['gender', 'first', 'last', 'dob', 'job', 'state', 'city']
#         data = data.drop(columns=drop_cols, errors='ignore')

#         # Feature engineering
#         if 'amt' in data.columns and 'city_pop' in data.columns:
#             data['amt_per_pop'] = data['amt'] / (data['city_pop'] + 1e-5)

#         if 'category' in data.columns:
#             print("Creating category-based statistical features...")
#             category_amt_mean = data.groupby('category')['amt'].transform('mean')
#             data['category_amt_mean'] = category_amt_mean
#             data['amt_above_mean'] = data['amt'] - data['category_amt_mean']

#         # Split into features and target
#         print("Splitting data into features and target...")
#         X = data.drop(columns=['is_fraud'])
#         y = data['is_fraud']

#         # Check for NaNs
#         if X.isna().any().any():
#             print("NaNs found in features:", X.isna().sum().to_dict())
#             raise ValueError("Input features contain NaNs. Please clean the data.")

#         # Train-test split
#         print("Splitting data into train and test sets...")
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#         print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")

#         # Apply SMOTE
#         print("Applying SMOTE...")
#         smote = SMOTE(sampling_strategy=0.25, random_state=42)
#         X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#         print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

#         # Scale features
#         print("Standardizing features...")
#         scaler = StandardScaler()
#         X_train_res = scaler.fit_transform(X_train_res)
#         X_test_scaled = scaler.transform(X_test)

#         # Base models
#         rf = RandomForestClassifier(n_estimators=300, max_depth=25, class_weight='balanced_subsample', random_state=42)
#         xgb = XGBClassifier(n_estimators=300, max_depth=10, scale_pos_weight=5, eval_metric='logloss', random_state=42)
#         lgbm = LGBMClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)

#         # Meta-model
#         meta_model = LogisticRegression(max_iter=1000)

#         # Stacking ensemble
#         print("Initializing stacked ensemble model...")
#         stack_model = StackingClassifier(
#             estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
#             final_estimator=meta_model,
#             passthrough=True,
#             n_jobs=-1
#         )

#         print("Performing cross-validation...")
#         cv_scores = cross_val_score(stack_model, X_train_res, y_train_res, cv=5, scoring='f1')
#         print(f"Cross-validation F1 scores: {cv_scores}")
#         print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

#         print("Training stacked model...")
#         stack_model.fit(X_train_res, y_train_res)

#         print("Testing multiple thresholds...")
#         thresholds = [round(t, 2) for t in np.arange(0.2, 0.81, 0.05)]
#         best_fbeta = 0
#         best_threshold = 0.2
#         best_metrics = {}
#         y_pred_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

#         for threshold in thresholds:
#             print(f"Evaluating threshold: {threshold}")
#             y_pred = (y_pred_proba >= threshold).astype(int)
#             metrics = {
#                 "accuracy": accuracy_score(y_test, y_pred),
#                 "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
#                 "recall": recall_score(y_test, y_pred, pos_label=1),
#                 "f1_score": f1_score(y_test, y_pred, pos_label=1),
#                 "fbeta_score": fbeta_score(y_test, y_pred, beta=2, pos_label=1)
#             }
#             print(f"Metrics at threshold {threshold}:")
#             for metric, value in metrics.items():
#                 print(f"{metric}: {value:.4f}")
#             print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#             if metrics['fbeta_score'] > best_fbeta:
#                 best_fbeta = metrics['fbeta_score']
#                 best_threshold = threshold
#                 best_metrics = metrics

#         print(f"Best threshold: {best_threshold} (Fβ-score: {best_fbeta:.4f})")
#         print("Final Confusion Matrix:\n", confusion_matrix(y_test, (y_pred_proba >= best_threshold).astype(int)))

#         return best_metrics

#     except Exception as e:
#         print(f"Error in train_is_fraud_model: {e}")
#         raise



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score

# def train_is_fraud_model(data: pd.DataFrame):
#     """
#     Trains a Random Forest model to predict fraud with cross-validation and threshold optimization.
    
#     Parameters:
#         data (pd.DataFrame): The dataset containing features and a binary target column named 'is_fraud'.
        
#     Returns:
#         dict: A dictionary containing model performance metrics.
#     """
#     try:
#         if 'is_fraud' not in data.columns:
#             raise ValueError("Dataset must contain an 'is_fraud' column as the target variable.")
        
#         # Check for valid binary labels
#         unique_labels = data['is_fraud'].unique()
#         if not set(unique_labels).issubset({0, 1}):
#             raise ValueError(f"'is_fraud' contains invalid labels: {unique_labels}. Expected [0, 1].")
        
#         # Drop low-importance feature
#         print("Dropping low-importance feature 'gender'...")
#         data = data.drop(columns=['gender'], errors='ignore')
        
#         # Splitting features and target
#         print("Splitting data into features and target...")
#         X = data.drop(columns=['is_fraud'])
#         y = data['is_fraud']
        
#         # Check for NaNs in X
#         if X.isna().any().any():
#             print("NaNs found in features:", X.isna().sum().to_dict())
#             raise ValueError("Input features contain NaNs. Please clean the data.")
        
#         # Splitting data into training and testing sets
#         print("Splitting data into train and test sets...")
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
#         # Print class distribution
#         print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
#         # Standardizing features
#         print("Standardizing features...")
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         # Initialize Random Forest with custom class weights
#         print("Initializing Random Forest model...")
#         model = RandomForestClassifier(
#             n_estimators=200,
#             max_depth=20,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             class_weight={0: 1, 1: 100},  # Emphasize fraud class
#             random_state=42
#         )
        
#         # Cross-validation
#         print("Performing cross-validation...")
#         cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
#         print(f"Cross-validation F1 scores: {cv_scores}")
#         print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
#         # Train the model
#         print("Training model...")
#         model.fit(X_train_scaled, y_train)
        
#         # Feature importance
#         print("Calculating feature importance...")
#         feature_importance = pd.DataFrame({
#             'feature': X.columns,
#             'importance': model.feature_importances_
#         }).sort_values('importance', ascending=False)
#         print("Feature Importance:\n", feature_importance)
        
#         # Test multiple thresholds
#         print("Testing multiple thresholds...")
#         thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
#         best_fbeta = 0
#         best_threshold = 0.1
#         best_metrics = {}
        
#         y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
#         for threshold in thresholds:
#             print(f"Evaluating threshold: {threshold}")
#             y_pred = (y_pred_proba >= threshold).astype(int)
#             metrics = {
#                 "accuracy": accuracy_score(y_test, y_pred),
#                 "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
#                 "recall": recall_score(y_test, y_pred, pos_label=1),
#                 "f1_score": f1_score(y_test, y_pred, pos_label=1),
#                 "fbeta_score": fbeta_score(y_test, y_pred, beta=2, pos_label=1)
#             }
#             print(f"Metrics at threshold {threshold}:")
#             for metric, value in metrics.items():
#                 print(f"{metric}: {value:.4f}")
#             print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#             if metrics['fbeta_score'] > best_fbeta:
#                 best_fbeta = metrics['fbeta_score']
#                 best_threshold = threshold
#                 best_metrics = metrics
        
#         # Final predictions with best threshold
#         print(f"Best threshold: {best_threshold} (Fβ-score: {best_fbeta:.4f})")
#         y_pred = (y_pred_proba >= best_threshold).astype(int)
#         print("Final Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
#         return best_metrics
    
#     except Exception as e:
#         print(f"Error in train_is_fraud_model: {e}")
#         raise

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
import warnings

# Suppress user warnings
warnings.filterwarnings('ignore', category=UserWarning)

def train_is_fraud_model(data: pd.DataFrame):
    """
    Trains an XGBoost model to predict fraud, using cross-validation, GridSearchCV, and threshold optimization.
    
    Parameters:
        data (pd.DataFrame): The dataset containing features and a binary target column named 'is_fraud'.
        
    Returns:
        dict: A dictionary containing model performance metrics.
    """
    try:
        if 'is_fraud' not in data.columns:
            raise ValueError("Dataset must contain an 'is_fraud' column as the target variable.")
        
        # Check for valid binary labels
        unique_labels = data['is_fraud'].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"'is_fraud' contains invalid labels: {unique_labels}. Expected [0, 1].")
        
        # Keep only top features
        print("Keeping top features: 'amt', 'amt_log', 'category', 'city_pop_log'...")
        keep_features = ['amt', 'amt_log', 'category', 'city_pop_log', 'is_fraud']
        data = data[keep_features]
        
        # Add interaction feature
        print("Adding interaction feature 'amt_category'...")
        data['amt_category'] = data['amt'] * data['category']
        
        # Splitting features and target
        print("Splitting data into features and target...")
        X = data.drop(columns=['is_fraud'])
        y = data['is_fraud']
        
        # Check for NaNs in X
        if X.isna().any().any():
            print("NaNs found in features:", X.isna().sum().to_dict())
            raise ValueError("Input features contain NaNs. Please clean the data.")
        
        # Splitting data into training and testing sets
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Print class distribution
        print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        # Standardizing features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize XGBoost
        print("Initializing XGBoost model...")
        model = XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        )
        
        # GridSearchCV for hyperparameter tuning
        print("Performing GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [5, 6],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [25, 50]
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        
        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        # Train the model
        print("Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Feature importance
        print("Calculating feature importance...")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Feature Importance:\n", feature_importance)
        
        # Test multiple thresholds
        print("Testing multiple thresholds...")
        thresholds = [0.2, 0.25, 0.3, 0.4, 0.5]
        best_f1 = 0
        best_threshold = 0.2
        best_metrics = {}
        
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        for threshold in thresholds:
            print(f"Evaluating threshold: {threshold}")
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                "recall": recall_score(y_test, y_pred, pos_label=1),
                "f1_score": f1_score(y_test, y_pred, pos_label=1),
                "fbeta_score": fbeta_score(y_test, y_pred, beta=2, pos_label=1)
            }
            print(f"Metrics at threshold {threshold}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
                best_metrics = metrics
        
        # Final predictions with best threshold
        print(f"Best threshold: {best_threshold} (F1-score: {best_f1:.4f})")
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        print("Final Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        return best_metrics
    
    except Exception as e:
        print(f"Error in train_is_fraud_model: {e}")
        raise