# Import required libraries
import mne
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from mne.decoding import CSP
from scipy import signal
import pandas as pd
import seaborn as sns
import os
import pickle
import hashlib
import time as time_module # Renamed due to AttributeError: 'numpy.ndarray' object has no attribute 'time'
import tkinter as tk
from tkinter import ttk, messagebox

# Import custom functions from the Inner Speech Dataset repository
from Inner_Speech_Dataset.Python_Processing.Data_extractions import (
    extract_data_from_subject,
    extract_data_multisubject,
)
from Inner_Speech_Dataset.Python_Processing.Data_processing import (
    select_time_window,
    transform_for_classificator,
)
from Inner_Speech_Dataset.Python_Processing.Data_processing import (
    filter_by_condition,
    filter_by_class,
)

# Set random seed for reproducibility
np.random.seed(23)

# Configure MNE to show only warnings (suppress info messages)
mne.set_log_level(verbose="warning")

# Suppress deprecation and future warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

# Create directories for figures and cache if they don't exist
figures_dir = "results/figures"
cache_dir = "cache"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

print("Setup complete! All libraries imported successfully.")

print("\n========== IMPROVED INNER SPEECH CLASSIFICATION ==========")
print("Goal: Improving classification accuracy beyond the baseline 50%")

# Set the hyperparameters for data loading and processing
# Directory containing the dataset (adjust this to your local path)
root_dir = "dataset"  # Path to the dataset

# Specify which type of data to use
# Options: "EEG" (brain activity), "exg" (physiological signals), or "baseline" (resting state)
datatype = "EEG"

# EEG sampling rate (256 samples per second)
fs = 256

# Select the time window of interest (in seconds)
# The cue appears at t=0, action period is between 1.5s and 3.5s
t_start = 1.5  # Start time after cue
t_end = 3.5  # End time after cue

# Function to create cache key based on parameters
def create_cache_key(params_dict):
    """Create a unique cache key based on parameters"""
    # Convert dictionary to string and hash it
    params_str = str(sorted(params_dict.items()))
    return hashlib.md5(params_str.encode()).hexdigest()

# Function to save data to cache
def save_to_cache(data, cache_key, description):
    """Save data to cache with given key"""
    cache_path = os.path.join(cache_dir, f"{description}_{cache_key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {description} to cache: {cache_path}")

# Function to load data from cache
def load_from_cache(cache_key, description):
    """Load data from cache with given key"""
    cache_path = os.path.join(cache_dir, f"{description}_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {description} from cache: {cache_path}")
        return data
    return None

# Function to extract frequency-domain features
def extract_frequency_features(X, fs, bands):
    """
    Extract power in different frequency bands for each channel.
    
    Parameters:
    -----------
    X : array, shape (n_trials, n_channels, n_times)
        EEG data
    fs : float
        Sampling frequency
    bands : list of tuples
        List of frequency bands (min_freq, max_freq, name)
        
    Returns:
    --------
    features : array, shape (n_trials, n_channels * n_bands)
        Frequency-domain features
    """
    n_trials, n_channels, n_times = X.shape
    n_bands = len(bands)
    
    # Initialize feature matrix
    features = np.zeros((n_trials, n_channels * n_bands))
    
    # For each trial and channel
    for i in range(n_trials):
        for j in range(n_channels):
            # Get the signal
            signal_data = X[i, j, :]
            
            # Calculate the power spectrum
            freqs, psd = signal.welch(signal_data, fs, nperseg=n_times//2)
            
            # Extract band power
            for k, (fmin, fmax, _) in enumerate(bands):
                # Find frequency indices within the band
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                
                # Calculate average power in the band
                if np.any(idx):
                    band_power = np.mean(psd[idx])
                else:
                    band_power = 0
                
                # Store the feature
                features[i, j * n_bands + k] = band_power
    
    return features

# Interactive UI for task selection
def launch_ui():
    """Launch a UI for selecting different mental tasks to compare"""
    root = tk.Tk()
    root.title("Mental Task Selection")
    root.geometry("800x600")  # Wider initial size
    root.minsize(600, 500)    # Set minimum size
    
    # Make the window resizable
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    # Configure style
    style = ttk.Style()
    style.configure('TLabel', font=('Arial', 12))
    style.configure('TButton', font=('Arial', 12))
    style.configure('TCheckbutton', font=('Arial', 11))
    style.configure('TRadiobutton', font=('Arial', 11))
    style.configure('TFrame', background='#f0f0f0')
    
    # Add a main frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.grid(row=0, column=0, sticky='nsew')  # Make it fill the entire window
    
    # Configure main_frame rows and columns to be resizable
    for i in range(8):  # Configure 8 rows
        main_frame.rowconfigure(i, weight=1 if i in [2, 4, 6] else 0)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Inner Speech Brain-Computer Interface", font=('Arial', 16, 'bold'))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Data type selection
    datatype_frame = ttk.LabelFrame(main_frame, text="Data Type", padding=10)
    datatype_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky='ew')
    
    datatype_var = tk.StringVar(value="EEG")
    datatype_options = ttk.Frame(datatype_frame)
    datatype_options.pack(fill='both', expand=True)
    datatype_options.columnconfigure((0,1,2), weight=1)
    
    ttk.Radiobutton(datatype_options, text="EEG (Brain)", variable=datatype_var, value="EEG").grid(row=0, column=0, padx=5, pady=2, sticky='w')
    ttk.Radiobutton(datatype_options, text="EXG (Muscle/Eye)", variable=datatype_var, value="exg").grid(row=0, column=1, padx=5, pady=2, sticky='w')
    ttk.Radiobutton(datatype_options, text="Baseline", variable=datatype_var, value="baseline").grid(row=0, column=2, padx=5, pady=2, sticky='w')
    
    # Subject selection
    subject_frame = ttk.LabelFrame(main_frame, text="Subjects", padding=10)
    subject_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')  # Allow vertical expansion
    
    subject_grid = ttk.Frame(subject_frame)
    subject_grid.pack(fill='both', expand=True)
    for i in range(5):  # 5 columns
        subject_grid.columnconfigure(i, weight=1)
    
    subject_vars = []
    for i in range(10):
        var = tk.BooleanVar(value=True)
        subject_vars.append(var)
        ttk.Checkbutton(subject_grid, text=f"Subject {i+1}", variable=var).grid(
            row=i//5, column=i%5, padx=10, pady=2, sticky='w')
    
    # Time window selection
    time_frame = ttk.LabelFrame(main_frame, text="Time Window (seconds)", padding=10)
    time_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky='ew')
    
    time_grid = ttk.Frame(time_frame)
    time_grid.pack(fill='both', expand=True)
    for i in range(4):
        time_grid.columnconfigure(i, weight=1)
    
    ttk.Label(time_grid, text="Start:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    t_start_var = tk.DoubleVar(value=1.5)
    t_start_entry = ttk.Entry(time_grid, textvariable=t_start_var, width=5)
    t_start_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
    
    ttk.Label(time_grid, text="End:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
    t_end_var = tk.DoubleVar(value=3.5)
    t_end_entry = ttk.Entry(time_grid, textvariable=t_end_var, width=5)
    t_end_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')
    
    # Mental task selection frames in a row that can expand horizontally
    tasks_container = ttk.Frame(main_frame)
    tasks_container.grid(row=4, column=0, columnspan=2, pady=10, sticky='nsew')
    tasks_container.columnconfigure(0, weight=1)
    tasks_container.columnconfigure(1, weight=1)
    
    # Mental task selection - First group
    task_frame1 = ttk.LabelFrame(tasks_container, text="Mental Task Group 1", padding=10)
    task_frame1.grid(row=0, column=0, padx=(0, 5), sticky='nsew')
    
    for i in range(2):
        task_frame1.columnconfigure(i, weight=1 if i == 1 else 0)
    
    ttk.Label(task_frame1, text="Condition:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    condition1_var = tk.StringVar(value="Inner")
    condition1_combo = ttk.Combobox(task_frame1, textvariable=condition1_var, 
                                   values=["Pronounced", "Inner", "Visualized"], state='readonly')
    condition1_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    
    ttk.Label(task_frame1, text="Direction:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
    direction1_var = tk.StringVar(value="Up")
    direction1_combo = ttk.Combobox(task_frame1, textvariable=direction1_var, 
                                   values=["Up", "Down", "Left", "Right"], state='readonly')
    direction1_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
    
    # Mental task selection - Second group
    task_frame2 = ttk.LabelFrame(tasks_container, text="Mental Task Group 2", padding=10)
    task_frame2.grid(row=0, column=1, padx=(5, 0), sticky='nsew')
    
    for i in range(2):
        task_frame2.columnconfigure(i, weight=1 if i == 1 else 0)
    
    ttk.Label(task_frame2, text="Condition:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    condition2_var = tk.StringVar(value="Inner")
    condition2_combo = ttk.Combobox(task_frame2, textvariable=condition2_var, 
                                   values=["Pronounced", "Inner", "Visualized"], state='readonly')
    condition2_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    
    ttk.Label(task_frame2, text="Direction:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
    direction2_var = tk.StringVar(value="Down")
    direction2_combo = ttk.Combobox(task_frame2, textvariable=direction2_var, 
                                   values=["Up", "Down", "Left", "Right"], state='readonly')
    direction2_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
    
    # Model selection
    model_frame = ttk.LabelFrame(main_frame, text="Classification Method", padding=10)
    model_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky='ew')
    
    model_grid = ttk.Frame(model_frame)
    model_grid.pack(fill='both', expand=True)
    model_grid.columnconfigure(0, weight=1)
    model_grid.columnconfigure(1, weight=1)
    
    model_var = tk.StringVar(value="ensemble")
    ttk.Radiobutton(model_grid, text="Ensemble (Best Performance)", variable=model_var, 
                   value="ensemble").grid(row=0, column=0, padx=10, pady=2, sticky='w')
    ttk.Radiobutton(model_grid, text="SVM", variable=model_var, 
                   value="svm").grid(row=0, column=1, padx=10, pady=2, sticky='w')
    ttk.Radiobutton(model_grid, text="LDA", variable=model_var, 
                   value="lda").grid(row=1, column=0, padx=10, pady=2, sticky='w')
    ttk.Radiobutton(model_grid, text="Random Forest", variable=model_var, 
                   value="rf").grid(row=1, column=1, padx=10, pady=2, sticky='w')
    
    # Results frame
    results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
    results_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky='ew')
    
    accuracy_var = tk.StringVar(value="Not computed yet")
    accuracy_label = ttk.Label(results_frame, text="Accuracy: ")
    accuracy_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    accuracy_value = ttk.Label(results_frame, textvariable=accuracy_var)
    accuracy_value.grid(row=0, column=1, padx=5, pady=5, sticky='w')
    
    # Function to run analysis
    def run_analysis():
        # Collect parameters
        datatype = datatype_var.get()
        t_start = t_start_var.get()
        t_end = t_end_var.get()
        subject_list = [i+1 for i in range(10) if subject_vars[i].get()]
        
        if not subject_list:
            messagebox.showerror("Error", "Please select at least one subject!")
            return
        
        # Define conditions and classes for comparison
        condition1 = condition1_var.get()
        direction1 = direction1_var.get()
        condition2 = condition2_var.get()
        direction2 = direction2_var.get()
        model_type = model_var.get()
        
        Conditions = [[condition1], [condition2]]
        Classes = [[direction1], [direction2]]
        
        # Update status
        accuracy_var.set("Computing...")
        root.update_idletasks()
        
        # Run analysis function
        try:
            accuracy = run_mental_task_analysis(subject_list, datatype, t_start, t_end, Conditions, Classes, model_type)
            accuracy_var.set(f"{accuracy:.2f}%")
            
            # Log to development log
            log_to_devlog(subject_list, datatype, t_start, t_end, Conditions, Classes, model_type, accuracy)
            
            # Show results message
            msg = f"Analysis complete!\n\nAccuracy: {accuracy:.2f}%\n\n"
            if accuracy < 60:
                msg += "This accuracy is relatively low. Consider trying different tasks or parameters."
            elif accuracy < 75:
                msg += "This accuracy is moderate. The model can detect some patterns but there's room for improvement."
            else:
                msg += "This accuracy is high! The model can reliably distinguish between these mental tasks."
            
            messagebox.showinfo("Results", msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            accuracy_var.set("Error")
            
    # Run button
    run_button = ttk.Button(main_frame, text="Run Analysis", command=run_analysis)
    run_button.grid(row=7, column=0, columnspan=2, pady=20)
    
    root.mainloop()

# Function to log results to the development log
def log_to_devlog(subject_list, datatype, t_start, t_end, Conditions, Classes, model_type, accuracy):
    """Add the results to the development log file"""
    log_file = "day1_devlog.md"
    
    try:
        with open(log_file, 'a') as f:
            f.write(f"\n\n### Analysis Results - {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Parameters:**\n")
            f.write(f"- Subjects: {subject_list}\n")
            f.write(f"- Data Type: {datatype}\n")
            f.write(f"- Time Window: {t_start}s to {t_end}s\n")
            f.write(f"- Group 1: {Classes[0][0]} direction in {Conditions[0][0]} condition\n")
            f.write(f"- Group 2: {Classes[1][0]} direction in {Conditions[1][0]} condition\n")
            f.write(f"- Model: {model_type}\n")
            f.write(f"- **Accuracy: {accuracy:.2f}%**\n")
            
            # Add comparison with baseline
            if Classes[0][0] == "Up" and Classes[1][0] == "Down" and Conditions[0][0] == "Inner" and Conditions[1][0] == "Inner":
                f.write(f"- Baseline for Up vs Down in Inner Speech: ~50%\n")
                f.write(f"- Improvement: {accuracy - 50:.2f}%\n")
            f.write("\n")
    except Exception as e:
        print(f"Warning: Could not write to development log: {str(e)}")

# Main function to run the analysis
def run_mental_task_analysis(subject_list, datatype, t_start, t_end, Conditions, Classes, model_type="ensemble"):
    """Run the complete mental task analysis with the specified parameters"""
    print("\n========== INNER SPEECH CLASSIFICATION ANALYSIS ==========")
    print(f"Analyzing {Classes[0][0]} vs {Classes[1][0]} in {Conditions[0][0]} vs {Conditions[1][0]} condition")
    
    # EEG sampling rate
    fs = 256
    
    # Create cache parameters for data loading
    cache_params = {
        'subject_list': subject_list,
        'datatype': datatype,
        't_start': t_start,
        't_end': t_end,
        'conditions': str(Conditions),
        'classes': str(Classes)
    }
    cache_key = create_cache_key(cache_params)

    # Try to load data from cache first
    cached_data = load_from_cache(cache_key, "multisubject_data")

    if cached_data is not None:
        # Use cached data
        X, Y = cached_data
        print("Using cached data for analysis")
    else:
        # Multi-subject data collection
        print("\nLoading and processing data from multiple subjects...")
        start_time = time_module.time()
        X_all_subjects = []
        Y_all_subjects = []

        for subject in subject_list:
            print(f"Processing subject {subject}...")
            # Load all trials for the selected subject
            try:
                # This combines data from all three experimental blocks
                X, Y = extract_data_from_subject("dataset", subject, datatype)
                
                # Extract only the relevant time window
                X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
                
                # Transform and filter the data to keep only trials of interest
                X_filtered, Y_filtered = transform_for_classificator(X, Y, Classes, Conditions)
                
                # Only add if we have trials for both classes
                if len(np.unique(Y_filtered)) > 1 and len(Y_filtered) > 0:
                    X_all_subjects.append(X_filtered)
                    Y_all_subjects.append(Y_filtered)
                    print(f"   Added {X_filtered.shape[0]} trials from subject {subject}")
                else:
                    print(f"   Skipping subject {subject} - insufficient class data")
                    
            except Exception as e:
                print(f"   Error processing subject {subject}: {e}")
                continue

        # Combine data from all subjects
        X = np.vstack(X_all_subjects)
        Y = np.concatenate(Y_all_subjects)
        
        # Save processed data to cache
        save_to_cache((X, Y), cache_key, "multisubject_data")
        
        processing_time = time_module.time() - start_time
        print(f"Data processing completed in {processing_time:.2f} seconds")

    print("\nCombined multi-subject data:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # Check how many trials we have for each class
    n_class1 = np.sum(Y == 0)
    n_class2 = np.sum(Y == 1)
    print(f"\nNumber of '{Classes[0][0]}' trials: {n_class1}")
    print(f"Number of '{Classes[1][0]}' trials: {n_class2}")

    # Feature extraction
    # Define frequency bands of interest
    bands = [
        (4, 8, 'Theta'),   # Theta band: 4-8 Hz (attention, memory)
        (8, 12, 'Alpha'),  # Alpha band: 8-12 Hz (inhibition, relaxation)
        (12, 30, 'Beta'),  # Beta band: 12-30 Hz (active thinking, motor planning)
        (30, 100, 'Gamma') # Gamma band: 30-100 Hz (complex cognitive processing)
    ]

    # Create cache parameters for feature extraction
    feature_cache_params = {
        'data_shape': X.shape,
        'bands': str(bands),
        'datatype': datatype,
        'conditions': str(Conditions),
        'classes': str(Classes)
    }
    feature_cache_key = create_cache_key(feature_cache_params)

    # Try to load features from cache
    cached_features = load_from_cache(feature_cache_key, "extracted_features")

    if cached_features is not None:
        # Use cached features
        X_freq, X_csp = cached_features
        print("Using cached features")
    else:
        print("\nExtracting frequency-domain features...")
        print(f"Frequency bands: {[(band[0], band[1], band[2]) for band in bands]}")
        start_time = time_module.time()
        
        # Extract frequency features
        X_freq = extract_frequency_features(X, fs, bands)
        print(f"Frequency features shape: {X_freq.shape}")

        # Apply Common Spatial Patterns (CSP) for spatial filtering
        print("\nApplying Common Spatial Patterns (CSP) filter...")
        
        # Reshape X for CSP which expects [n_trials, n_channels, n_times]
        n_components = 6  # Number of CSP components to use
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

        # Make an explicit copy of X to double precision to fix the ValueError
        X_for_csp = X.astype(np.float64)
        
        # Apply CSP transform
        X_csp = csp.fit_transform(X_for_csp, Y)
        print(f"CSP features shape: {X_csp.shape}")
        
        # Save features to cache
        save_to_cache((X_freq, X_csp), feature_cache_key, "extracted_features")
        
        feature_time = time_module.time() - start_time
        print(f"Feature extraction completed in {feature_time:.2f} seconds")

    # Combine features
    X_combined = np.hstack([X_freq, X_csp])
    print(f"Combined features shape: {X_combined.shape}")

    # Feature selection
    # Feature selection cache parameters
    selection_cache_params = {
        'feature_shape': X_combined.shape,
        'feature_key': feature_cache_key
    }
    selection_cache_key = create_cache_key(selection_cache_params)

    # Try to load selected features from cache
    cached_selection = load_from_cache(selection_cache_key, "selected_features")

    if cached_selection is not None:
        # Use cached selection
        X_selected, selected_indices = cached_selection
        print("Using cached feature selection")
    else:
        # Feature selection to reduce dimensionality
        print("\nApplying feature selection to identify most discriminative features...")
        start_time = time_module.time()
        
        selector = SelectKBest(score_func=f_classif, k=min(50, X_combined.shape[1]))
        X_selected = selector.fit_transform(X_combined, Y)
        print(f"Selected features shape: {X_selected.shape}")

        # Get indices of selected features for interpretation
        selected_indices = selector.get_support(indices=True)
        print(f"Number of selected features: {len(selected_indices)}")
        
        # Save selected features to cache
        save_to_cache((X_selected, selected_indices), selection_cache_key, "selected_features")
        
        selection_time = time_module.time() - start_time
        print(f"Feature selection completed in {selection_time:.2f} seconds")

    # Split the selected features into training and testing sets
    X_train_selected, X_test_selected, y_train, y_test = train_test_split(
        X_selected, Y, test_size=0.3, random_state=42
    )

    print(f"\nTraining set: {X_train_selected.shape[0]} trials")
    print(f"Testing set: {X_test_selected.shape[0]} trials")

    # Model training cache parameters
    model_cache_params = {
        'selection_key': selection_cache_key,
        'train_shape': X_train_selected.shape,
        'model_type': model_type
    }
    model_cache_key = create_cache_key(model_cache_params)

    # Try to load models from cache
    cached_models = load_from_cache(model_cache_key, "trained_models")

    if cached_models is not None:
        # Use cached models
        models = cached_models
        print("Using cached trained models")
    else:
        # Train models
        print("\nTraining classification models...")
        start_time = time_module.time()

        models = {}
        
        # Cross-validation setup
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        if model_type in ["ensemble", "svm"]:
            # SVM
            print("\nTraining SVM classifier...")
            svm_param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
            svm_clf = SVC(probability=True, random_state=42)
            svm_grid_search = GridSearchCV(estimator=svm_clf, param_grid=svm_param_grid, 
                                        cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
            svm_grid_search.fit(X_train_selected, y_train)
            print(f"Best SVM parameters: {svm_grid_search.best_params_}")
            print(f"SVM cross-validation accuracy: {svm_grid_search.best_score_:.2%}")
            models['svm'] = svm_grid_search
        
        if model_type in ["ensemble", "rf"]:
            # Random Forest
            print("\nTraining Random Forest classifier...")
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            rf_clf = RandomForestClassifier(random_state=42)
            rf_grid_search = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, 
                                        cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
            rf_grid_search.fit(X_train_selected, y_train)
            print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")
            print(f"Random Forest cross-validation accuracy: {rf_grid_search.best_score_:.2%}")
            models['rf'] = rf_grid_search
        
        if model_type in ["ensemble", "lda"]:
            # LDA
            print("\nTraining LDA classifier...")
            lda_param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            }
            lda_clf = LDA()
            lda_grid_search = GridSearchCV(estimator=lda_clf, param_grid=lda_param_grid, 
                                        cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
            lda_grid_search.fit(X_train_selected, y_train)
            print(f"Best LDA parameters: {lda_grid_search.best_params_}")
            print(f"LDA cross-validation accuracy: {lda_grid_search.best_score_:.2%}")
            models['lda'] = lda_grid_search
        
        if model_type == "ensemble":
            # Ensemble
            print("\nTraining Ensemble classifier...")
            estimators = [
                ('svm', models['svm'].best_estimator_),
                ('rf', models['rf'].best_estimator_),
                ('lda', models['lda'].best_estimator_)
            ]
            
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train_selected, y_train)
            models['ensemble'] = ensemble
        
        # Save models to cache
        save_to_cache(models, model_cache_key, "trained_models")
        
        model_time = time_module.time() - start_time
        print(f"Model training completed in {model_time:.2f} seconds")

    # Evaluate on test set
    print("\nEvaluating classifier on test data...")
    
    # Use the requested model for prediction
    if model_type == "ensemble" and "ensemble" in models:
        y_pred = models['ensemble'].predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Ensemble Test Accuracy: {accuracy:.2f}%")
    elif model_type == "svm" and "svm" in models:
        y_pred = models['svm'].predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"SVM Test Accuracy: {accuracy:.2f}%")
    elif model_type == "rf" and "rf" in models:
        y_pred = models['rf'].predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Random Forest Test Accuracy: {accuracy:.2f}%")
    elif model_type == "lda" and "lda" in models:
        y_pred = models['lda'].predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"LDA Test Accuracy: {accuracy:.2f}%")
    else:
        raise ValueError(f"Requested model {model_type} not available!")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=Classes, yticklabels=Classes)
    plt.title(f"{model_type.upper()} Model Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Direction", fontsize=14)
    plt.ylabel("True Direction", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"confusion_matrix_{model_type}_{Classes[0][0]}_{Classes[1][0]}.png"))
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=Classes[0] + Classes[1]))

    print(f"\nAnalysis Complete: {accuracy:.2f}% accuracy")
    return accuracy

# Main execution
if __name__ == "__main__":
    # Launch the UI for task selection
    print("Setup complete! All libraries imported successfully.")
    print("\n========== IMPROVED INNER SPEECH CLASSIFICATION ==========")
    print("Goal: Improving classification accuracy beyond the baseline 50%")
    print("\nLaunching interactive UI for mental task selection...")
    launch_ui()