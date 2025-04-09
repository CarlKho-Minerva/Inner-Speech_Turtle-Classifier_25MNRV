# day2.py: Systematic Pairwise Classification Analysis

# --- Script Overview ---
# Goal: Systematically test classification accuracy and other metrics for all 
#       unique pairwise combinations of mental states (Condition x Class) 
#       using the enhanced pipeline in day2_analysis_pipeline.py.
# Rationale: Map the discriminability landscape of the dataset to identify
#            promising contrasts for BCI control and understand limitations.
# Output: Appends detailed results (metrics, time, balance) for each pair to 
#         day2_devlog.md and saves a final summary to day2_pairwise_results.txt.
#
# --- Key Considerations & Limitations (See day2_devlog.md for details) ---
# 1. Fixed Pipeline: Uses the *same* feature extraction (Freq+CSP), feature 
#    selection (k=50), and model tuning strategy for all 66 pairs. 
#    This may not be optimal for every specific contrast.
# 2. Metrics Focus: Reports Accuracy, Precision, Recall, F1. Consider AUC for 
#    imbalanced pairs.
# 3. Neuroscience Caveats: Assumes features capture relevant differences for 
#    all pairs and that the time window represents pure states.
# 4. No Statistical Testing: Does not perform permutation tests to assess if
#    performance is significantly above chance.
# -----------------------------------

import itertools
import time
import numpy as np
import os
import datetime # Added for timestamping
import pandas as pd # Added for results summary

# --- Import the enhanced analysis function --- 
from day2_analysis_pipeline import run_pairwise_analysis_enhanced

print("========== DAY 2: SYSTEMATIC PAIRWISE CLASSIFICATION (ENHANCED PIPELINE) ==========")

# --- Configuration ---
subject_list = list(range(1, 11))  # Subjects 1-10
datatype = "EEG"
t_start = 1.5  # Start time after cue (seconds)
t_end = 3.5  # End time after cue (seconds)
# model_type_to_run is handled within the enhanced function (always ensemble)
devlog_file = "day2_devlog.md" # Define devlog filename
results_file = "day2_pairwise_results.csv" # Changed to CSV for richer data

# Define all possible conditions and classes
conditions = ["Inner", "Pron", "Vis"]
classes = ["Up", "Down", "Left", "Right"]

# Generate all 12 unique mental states (Condition, Class)
all_states = list(itertools.product(conditions, classes))
print(f"Total number of unique mental states: {len(all_states)}")
print(all_states)

# Generate all unique pairwise combinations of these states
state_pairs = list(itertools.combinations(all_states, 2))
print(f"Total number of unique pairwise comparisons: {len(state_pairs)}")

# --- Analysis Loop ---
results_list = [] # Store results as a list of dictionaries

total_pairs = len(state_pairs)
start_time_total = time.time()

# Add a timestamp to the devlog when starting the run
with open(devlog_file, "a") as f:
    f.write(f"\n**Run Started (Enhanced Pipeline): {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**\n")

for i, pair in enumerate(state_pairs):
    state1_cond, state1_cls = pair[0]
    state2_cond, state2_cls = pair[1]

    pair_name = f"{state1_cond}-{state1_cls}_vs_{state2_cond}-{state2_cls}"
    print(f"\n--- Running Analysis {i+1}/{total_pairs}: {pair_name} ---")

    # Format conditions and classes for the analysis function
    current_conditions = [[state1_cond], [state2_cond]]
    current_classes = [[state1_cls], [state2_cls]]

    try:
        start_time_pair = time.time()
        # --- Call the enhanced analysis function --- 
        metrics = run_pairwise_analysis_enhanced(
            subject_list=subject_list,
            datatype=datatype,
            t_start=t_start,
            t_end=t_end,
            Conditions=current_conditions,
            Classes=current_classes,
            pair_name=pair_name # Pass pair name for context
        )
        
        end_time_pair = time.time()
        time_taken = end_time_pair - start_time_pair
        print(f"--- Time for {pair_name}: {time_taken:.2f} seconds ---")

        # Store and log the result
        result_log_entry = f"- {pair_name}: "
        result_dict = {"pair_name": pair_name, "time_seconds": time_taken}

        if metrics is not None:
            result_dict.update(metrics) # Add all metrics from the returned dict
            result_log_entry += f"Acc={metrics['accuracy']:.1f}%, P={metrics['precision']:.1f}%, R={metrics['recall']:.1f}%, F1={metrics['f1_score']:.1f}% (Bal={metrics['balance_ratio']:.2f}, N={metrics['n_trials_total']}) (Time: {time_taken:.2f}s)"
        else:
            result_dict.update({
                "accuracy": "N/A", "precision": "N/A", "recall": "N/A", "f1_score": "N/A",
                "n_trials_total": "N/A", "n_trials_class0": "N/A", "n_trials_class1": "N/A",
                "balance_ratio": "N/A", "error": "No data or error in analysis"
            })
            result_log_entry += "Error or N/A in analysis" + f" (Time: {time_taken:.2f}s)"
            print(f"--- Analysis for {pair_name} did not return metrics. ---")

        results_list.append(result_dict)

        # Append result to devlog immediately
        with open(devlog_file, "a") as f:
            f.write(result_log_entry + "\n")

    except Exception as e:
        error_msg = f"Error: {e}"
        print(f"!!! ERROR during analysis for {pair_name}: {e} !!!")
        # Append error to devlog immediately
        with open(devlog_file, "a") as f:
            f.write(f"- {pair_name}: {error_msg}\n")
        # Store error in results list
        results_list.append({"pair_name": pair_name, "error": error_msg})

end_time_total = time.time()
total_duration_minutes = (end_time_total - start_time_total) / 60
print(f"\n========== TOTAL ANALYSIS COMPLETE ==========")
print(f"Total time for {total_pairs} pairs: {total_duration_minutes:.2f} minutes")

# Add completion timestamp to the devlog
with open(devlog_file, "a") as f:
    f.write(f"**Run Finished (Enhanced Pipeline): {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Total Duration: {total_duration_minutes:.2f} min)**\n")

# --- Save Summary of Results to CSV ---
print(f"\nSaving detailed results summary to {results_file}...")
if results_list:
    results_df = pd.DataFrame(results_list)
    # Reorder columns for better readability
    cols_order = ['pair_name', 'accuracy', 'f1_score', 'precision', 'recall', 'balance_ratio', 'n_trials_total', 'n_trials_class0', 'n_trials_class1', 'time_seconds', 'error']
    # Filter out columns that might not exist if there were errors
    cols_order = [col for col in cols_order if col in results_df.columns]
    results_df = results_df[cols_order]
    # Sort by accuracy (descending), handling potential non-numeric values
    results_df['accuracy_numeric'] = pd.to_numeric(results_df['accuracy'], errors='coerce')
    results_df = results_df.sort_values(by='accuracy_numeric', ascending=False, na_position='last')
    results_df = results_df.drop(columns=['accuracy_numeric'])
    
    try:
        results_df.to_csv(results_file, index=False, float_format='%.2f')
        print(f"Results summary saved successfully.")
    except Exception as e:
        print(f"Error saving results summary to CSV: {e}")
else:
    print("No results were generated to save.")

print(f"Live results appended to {devlog_file}")
