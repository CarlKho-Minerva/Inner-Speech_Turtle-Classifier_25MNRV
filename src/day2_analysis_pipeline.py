# day2_analysis_pipeline.py
# Contains the enhanced analysis pipeline specifically for day2.py

import mne
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from mne.decoding import CSP
from scipy import signal
import seaborn as sns
import os
import pickle
import hashlib
import time as time_module

# Import necessary functions from the original dataset processing scripts
# Assuming these are accessible in the environment/path
try:
    from Inner_Speech_Dataset.Python_Processing.Data_extractions import extract_data_from_subject
    from Inner_Speech_Dataset.Python_Processing.Data_processing import select_time_window, transform_for_classificator
except ImportError:
    print("Warning: Could not import functions from Inner_Speech_Dataset. Ensure the package is installed or path is correct.")
    # Define dummy functions if import fails, to allow script structure to load
    def extract_data_from_subject(*args, **kwargs): raise NotImplementedError("Inner_Speech_Dataset not found")
    def select_time_window(*args, **kwargs): raise NotImplementedError("Inner_Speech_Dataset not found")
    def transform_for_classificator(*args, **kwargs): raise NotImplementedError("Inner_Speech_Dataset not found")

# --- Configuration & Setup ---
mne.set_log_level(verbose="warning")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

figures_dir = "figures" # Ensure this exists if plotting is re-enabled
cache_dir = "cache"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

fs = 256 # EEG sampling rate

# --- Caching Functions (Copied from day1/day1_ui) ---
def create_cache_key(params_dict):
    params_str = str(sorted(params_dict.items()))
    return hashlib.md5(params_str.encode()).hexdigest()

def save_to_cache(data, cache_key, description):
    cache_path = os.path.join(cache_dir, f"{description}_{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {description} to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving {description} to cache: {e}")

def load_from_cache(cache_key, description):
    cache_path = os.path.join(cache_dir, f"{description}_{cache_key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {description} from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"Error loading {description} from cache: {e}")
            # Attempt to delete corrupted cache file
            try:
                os.remove(cache_path)
                print(f"Removed potentially corrupted cache file: {cache_path}")
            except OSError as oe:
                print(f"Error removing corrupted cache file: {oe}")
    return None

# --- Feature Extraction Function (Copied from day1) ---
def extract_frequency_features(X, fs, bands):
    n_trials, n_channels, n_times = X.shape
    n_bands = len(bands)
    features = np.zeros((n_trials, n_channels * n_bands))
    for i in range(n_trials):
        for j in range(n_channels):
            signal_data = X[i, j, :]
            freqs, psd = signal.welch(signal_data, fs, nperseg=min(n_times, 256)) # Use nperseg=256 or n_times if shorter
            for k, (fmin, fmax, _) in enumerate(bands):
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                if np.any(idx):
                    band_power = np.mean(psd[idx])
                else:
                    band_power = 0
                features[i, j * n_bands + k] = band_power
    return features

# --- Enhanced Analysis Function for Day 2 ---
def run_pairwise_analysis_enhanced(subject_list, datatype, t_start, t_end, Conditions, Classes, pair_name):
    """
    Runs enhanced pairwise classification analysis for day2.py.

    Includes:
    - Caching for data loading, feature extraction, selection, and models.
    - Class balance check and reporting.
    - Feature Extraction: Frequency Bands + CSP.
    - Feature Selection: SelectKBest (k=50).
    - Classification: Tuned Ensemble (SVM, RF, LDA) via GridSearchCV.
    - Evaluation: Returns Accuracy, Precision, Recall, F1-score.
    """
    print(f"\nStarting Enhanced Analysis for: {pair_name}")
    print(f"Subjects: {subject_list}, Time: {t_start}-{t_end}s")

    # --- 1. Data Loading & Preprocessing ---
    data_cache_params = {
        'subject_list': subject_list, 'datatype': datatype, 't_start': t_start,
        't_end': t_end, 'conditions': str(Conditions), 'classes': str(Classes)
    }
    data_cache_key = create_cache_key(data_cache_params)
    cached_data = load_from_cache(data_cache_key, "multisubject_data")

    if cached_data is not None:
        X, Y = cached_data
        print("Using cached multi-subject data")
    else:
        print("Loading and processing data from multiple subjects...")
        start_time = time_module.time()
        X_all_subjects, Y_all_subjects = [], []
        for subject in subject_list:
            print(f"Processing subject {subject}...")
            try:
                # Load data (adjust root_dir if necessary, assuming 'dataset' is accessible)
                X_subj, Y_subj = extract_data_from_subject("dataset", subject, datatype)
                X_subj = select_time_window(X=X_subj, t_start=t_start, t_end=t_end, fs=fs)
                X_filtered, Y_filtered = transform_for_classificator(X_subj, Y_subj, Classes, Conditions)
                if len(np.unique(Y_filtered)) > 1 and len(Y_filtered) > 0:
                    X_all_subjects.append(X_filtered)
                    Y_all_subjects.append(Y_filtered)
                    print(f"   Added {X_filtered.shape[0]} trials from subject {subject}")
                else:
                    print(f"   Skipping subject {subject} - insufficient class data for this pair")
            except Exception as e:
                print(f"   Error processing subject {subject}: {e}")
                continue
        
        if not X_all_subjects:
             print("Error: No subjects had sufficient data for both classes in this pair.")
             return None # Return None if no data could be loaded

        X = np.vstack(X_all_subjects)
        Y = np.concatenate(Y_all_subjects)
        save_to_cache((X, Y), data_cache_key, "multisubject_data")
        print(f"Data processing completed in {time_module.time() - start_time:.2f} seconds")

    print(f"Combined data shape: X={X.shape}, Y={Y.shape}")

    # **Enhancement: Class Balance Check**
    class_counts = np.unique(Y, return_counts=True)
    print("Class Balance Check:")
    if len(class_counts[0]) == 2:
        print(f"  Class 0 ({Classes[0][0]} in {Conditions[0][0]}): {class_counts[1][0]} trials")
        print(f"  Class 1 ({Classes[1][0]} in {Conditions[1][0]}): {class_counts[1][1]} trials")
        balance_ratio = min(class_counts[1]) / max(class_counts[1]) if max(class_counts[1]) > 0 else 0
        print(f"  Balance Ratio (min/max): {balance_ratio:.2f}")
        if balance_ratio < 0.7: # Arbitrary threshold for warning
             print("  Warning: Significant class imbalance detected. Accuracy might be less informative.")
    else:
        print("  Warning: Expected 2 classes, but found:", class_counts[0])
        return None # Cannot proceed with binary classification pipeline

    # --- 2. Feature Extraction (Freq + CSP) ---
    bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 100, 'Gamma')]
    feature_cache_params = {'data_key': data_cache_key, 'bands': str(bands), 'csp_components': 6}
    feature_cache_key = create_cache_key(feature_cache_params)
    cached_features = load_from_cache(feature_cache_key, "extracted_features_day2")

    if cached_features is not None:
        X_freq, X_csp = cached_features
        print("Using cached features (Freq+CSP)")
    else:
        print("Extracting features (Frequency Bands + CSP)...")
        start_time = time_module.time()
        # Freq
        X_freq = extract_frequency_features(X, fs, bands)
        # CSP
        n_components = 6
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        X_for_csp = X.astype(np.float64) # Ensure float64 for CSP
        X_csp = csp.fit_transform(X_for_csp, Y)
        save_to_cache((X_freq, X_csp), feature_cache_key, "extracted_features_day2")
        print(f"Feature extraction completed in {time_module.time() - start_time:.2f} seconds")

    X_combined = np.hstack([X_freq, X_csp])
    print(f"Combined features shape: {X_combined.shape}")

    # --- 3. Feature Selection (SelectKBest k=50) ---
    # Note: Keeping k=50 fixed for consistency across pairs as discussed.
    k_features = min(50, X_combined.shape[1])
    selection_cache_params = {'feature_key': feature_cache_key, 'k': k_features}
    selection_cache_key = create_cache_key(selection_cache_params)
    cached_selection = load_from_cache(selection_cache_key, "selected_features_day2")

    if cached_selection is not None:
        X_selected = cached_selection
        print(f"Using cached selected features (k={k_features})")
    else:
        print(f"Applying feature selection (SelectKBest, k={k_features})...")
        start_time = time_module.time()
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_selected = selector.fit_transform(X_combined, Y)
        save_to_cache(X_selected, selection_cache_key, "selected_features_day2")
        print(f"Feature selection completed in {time_module.time() - start_time:.2f} seconds")
    print(f"Selected features shape: {X_selected.shape}")

    # --- 4. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, Y, test_size=0.3, random_state=42, stratify=Y # Stratify helps with balance
    )
    print(f"Train/Test Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # --- 5. Model Training (Tuned Ensemble) ---
    model_cache_params = {'selection_key': selection_cache_key, 'train_shape': X_train.shape}
    model_cache_key = create_cache_key(model_cache_params)
    cached_models = load_from_cache(model_cache_key, "trained_models_day2")

    if cached_models is not None:
        ensemble_model = cached_models
        print("Using cached trained ensemble model")
    else:
        print("Training classification models (SVM, RF, LDA) with GridSearchCV...")
        start_time = time_module.time()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        models = {}

        # Define grids (consider smaller grids if runs are too long)
        svm_param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear'], 'probability': [True]}
        rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
        lda_param_grid = {'solver': ['svd', 'lsqr'], 'shrinkage': [None, 'auto', 0.5]} # Removed 'eigen' as it needs shrinkage=None

        # SVM
        svm_clf = SVC(random_state=42)
        svm_grid = GridSearchCV(svm_clf, svm_param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
        svm_grid.fit(X_train, y_train)
        models['svm'] = svm_grid.best_estimator_
        print(f"  Best SVM CV Score: {svm_grid.best_score_:.2%}")

        # Random Forest
        rf_clf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf_clf, rf_param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
        rf_grid.fit(X_train, y_train)
        models['rf'] = rf_grid.best_estimator_
        print(f"  Best RF CV Score: {rf_grid.best_score_:.2%}")

        # LDA
        lda_clf = LDA()
        lda_grid = GridSearchCV(lda_clf, lda_param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
        lda_grid.fit(X_train, y_train)
        models['lda'] = lda_grid.best_estimator_
        print(f"  Best LDA CV Score: {lda_grid.best_score_:.2%}")

        # Ensemble
        print("Training Ensemble classifier (soft voting)...")
        estimators = list(models.items())
        ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
        ensemble_model.fit(X_train, y_train)

        save_to_cache(ensemble_model, model_cache_key, "trained_models_day2")
        print(f"Model training completed in {time_module.time() - start_time:.2f} seconds")

    # --- 6. Evaluation ---
    print("Evaluating ensemble model on test data...")
    y_pred = ensemble_model.predict(X_test)

    # **Enhancement: Calculate Richer Metrics**
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100

    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Test Precision: {precision:.2f}%")
    print(f"  Test Recall: {recall:.2f}%")
    print(f"  Test F1-Score: {f1:.2f}%")

    # Optional: Print classification report for details
    # print("
    # Classification Report (Test Set):")
    # target_names = [f"{Classes[0][0]}_{Conditions[0][0]}", f"{Classes[1][0]}_{Conditions[1][0]}"]
    # print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Return dictionary of metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "n_trials_total": X.shape[0],
        "n_trials_class0": class_counts[1][0] if len(class_counts[0])==2 else 'N/A',
        "n_trials_class1": class_counts[1][1] if len(class_counts[0])==2 else 'N/A',
        "balance_ratio": balance_ratio if len(class_counts[0])==2 else 'N/A'
    }

    return metrics

# --- Statistical Significance Note ---
# Calculating statistical significance (e.g., via permutation testing) for each of
# the 66 pairs within this loop would be extremely time-consuming.
# It's recommended to run significance tests as a separate, focused analysis step
# on specific pairs of interest identified from these initial results.
# Example (conceptual):
# from sklearn.model_selection import permutation_test_score
# score, permutation_scores, pvalue = permutation_test_score(
#     ensemble_model, X_selected, Y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=-1
# )
# print(f"P-value (Permutation Test): {pvalue:.4f}")
