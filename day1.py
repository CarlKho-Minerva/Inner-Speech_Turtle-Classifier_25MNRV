# Import required libraries
import mne
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Import custom functions from the Inner Speech Dataset repository
from Inner_Speech_Dataset.Python_Processing.Data_extractions import (
    extract_data_from_subject,
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

print("Setup complete! All libraries imported successfully.")

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

# Subject to analyze (1-10)
N_S = 1  # We'll start with subject 1

print(
    f"Parameters set: Using {datatype} data from subject {N_S}, analyzing time window {t_start}s to {t_end}s"
)

# Load all trials for the selected subject
# This combines data from all three experimental blocks
X, Y = extract_data_from_subject(root_dir, N_S, datatype)

print("Data successfully loaded!")
print("\nX (EEG data) shape:", X.shape)
print("  - Number of trials:", X.shape[0])
print("  - Number of EEG channels:", X.shape[1])
print("  - Number of time points per trial:", X.shape[2])
print("  - Total duration per trial:", X.shape[2] / fs, "seconds")

print("\nY (labels) shape:", Y.shape)
print("  - Number of trials:", Y.shape[0])
print("  - Label information columns:", Y.shape[1])

# Let's look at the first 5 trials and their labels
print("\nFirst 5 trial labels:")
print("Trial\tTimestamp\tDirection\tCondition\tSession")
for i in range(5):
    print(
        f"{i+1}\t{Y[i,0]}\t\t{['Up', 'Down', 'Right', 'Left'][Y[i,1]]}\t\t{['Pronounced', 'Inner', 'Visualized'][Y[i,2]]}\t\t{Y[i,3]}"
    )

# Check how many trials we have of each condition and direction
conditions = ["Pronounced", "Inner Speech", "Visualized"]
directions = ["Up", "Down", "Right", "Left"]

print("\nNumber of trials per condition:")
for i, condition in enumerate(conditions):
    condition_count = np.sum(Y[:, 2] == i)
    print(f"{condition}: {condition_count}")

print("\nNumber of trials per direction:")
for i, direction in enumerate(directions):
    direction_count = np.sum(Y[:, 1] == i)
    print(f"{direction}: {direction_count}")

# Extract only the relevant time window from each trial
# X goes from [trials × channels × full_time_points] to [trials × channels × selected_time_points]
X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)

print("Time window selection complete!")
print("\nNew X (EEG data) shape:", X.shape)
print("  - Number of trials:", X.shape[0])
print("  - Number of EEG channels:", X.shape[1])
print("  - Number of time points per trial:", X.shape[2])
print("  - Selected duration per trial:", X.shape[2] / fs, "seconds")

# Calculate how many data points were selected
total_data_points = X.shape[0] * X.shape[1] * X.shape[2]
print(f"\nTotal data points in selected window: {total_data_points:,}")

# First, let's reload the original data for our subject
X, Y = extract_data_from_subject(root_dir, N_S, datatype)

# Extract only the relevant time window
X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)

# Define which conditions and classes we want to compare
# We'll compare "Up" vs "Down" in the "Inner Speech" condition
Conditions = [["Inner"], ["Inner"]]  # Both groups use Inner Speech condition
Classes = [["Up"], ["Down"]]  # First group is "Up", second is "Down"

print("Filtering data to compare:")
print(f"- Group 1: {Classes[0][0]} direction in {Conditions[0][0]} condition")
print(f"- Group 2: {Classes[1][0]} direction in {Conditions[1][0]} condition")

# Transform and filter the data to keep only trials of interest
X, Y = transform_for_classificator(X, Y, Classes, Conditions)

print("\nAfter filtering:")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("\nY now contains simple binary labels:")
print("- 0 represents:", Classes[0][0])
print("- 1 represents:", Classes[1][0])

# Check how many trials we have for each class
n_up = np.sum(Y == 0)
n_down = np.sum(Y == 1)
print(f"\nNumber of '{Classes[0][0]}' trials: {n_up}")
print(f"Number of '{Classes[1][0]}' trials: {n_down}")

# Choose a central electrode to visualize (e.g., Cz which is typically electrode 65)
electrode_idx = 68  # This might need adjustment based on the actual channel layout

# Calculate means for each class
up_trials = X[Y == 0]
down_trials = X[Y == 1]

up_mean = np.mean(up_trials[:, electrode_idx, :], axis=0)
down_mean = np.mean(down_trials[:, electrode_idx, :], axis=0)

# Create a time vector (in seconds)
time = np.linspace(t_start, t_end, X.shape[2])
# Plot the average signals
plt.figure(figsize=(12, 6))

# Calculate standard deviation for both classes to show variability
up_std = np.std(up_trials[:, electrode_idx, :], axis=0)
down_std = np.std(down_trials[:, electrode_idx, :], axis=0)

# Plot mean lines
plt.plot(time, up_mean, "b-", linewidth=2, label='Thinking "Up"')
plt.plot(time, down_mean, "r-", linewidth=2, label='Thinking "Down"')

# Add shaded areas for standard deviation to show variability
plt.fill_between(time, up_mean - up_std, up_mean + up_std, color="blue", alpha=0.2)
plt.fill_between(
    time, down_mean - down_std, down_mean + down_std, color="red", alpha=0.2
)

# Add horizontal line at zero
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.grid(True, alpha=0.3)

# Add title and labels
plt.title(
    f"Average EEG Signal at Central Electrode (Channel {electrode_idx})", fontsize=16
)
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("Amplitude (microvolts)", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

print("The plot shows how brain activity differs when thinking 'Up' versus 'Down'.")
print("Notice the subtle differences in the patterns between the two mental tasks.")

# Reshape X from [trials × channels × time points] to [trials × features]
# where features = channels * time points
n_trials = X.shape[0]
n_features = X.shape[1] * X.shape[2]  # channels * time points

# Flatten each trial into a single feature vector
X_features = X.reshape(n_trials, n_features)

print(f"Data reshaped from {X.shape} to {X_features.shape}")
print(f"Each trial now has {n_features:,} features")

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_features, Y, test_size=0.3, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} trials")
print(f"Testing set: {X_test.shape[0]} trials")

# Create a pipeline that standardizes the data and then applies SVM
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ]
)

# Train the model
print("Training the classifier...")
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nClassification accuracy: {accuracy:.2f}%")

# Create and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix", fontsize=16)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Up", "Down"])
plt.yticks(tick_marks, ["Up", "Down"])
plt.xlabel("Predicted Direction", fontsize=14)
plt.ylabel("True Direction", fontsize=14)

# Add text annotations to the confusion matrix
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.tight_layout()
plt.show()

print("\nThe confusion matrix shows:")
print(f"- True Up, predicted Up: {cm[0,0]} trials")
print(f"- True Up, predicted Down: {cm[0,1]} trials")
print(f"- True Down, predicted Up: {cm[1,0]} trials")
print(f"- True Down, predicted Down: {cm[1,1]} trials")