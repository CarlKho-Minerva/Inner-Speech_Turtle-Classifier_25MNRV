Clone [Inner_Speech_Data](https://github.com/N-Nieto/Inner_Speech_Dataset)

Install python modules:

- mne
- numpy
- scikit-learn
- ipywidgets

# Inner Speech BCI - Tutorial Guide

## Introduction
Welcome to the Inner Speech Brain-Computer Interface (BCI) tutorial! This project allows you to explore how brain signals captured through electroencephalography (EEG) can be used to classify different types of inner speech - the internal dialogue that occurs when you think about words without saying them aloud.

## Code Structure and Components: Line-by-Line Explanation

### Data Loading and Preprocessing
The tutorial loads preprocessed EEG data from participants who performed mental tasks involving directional commands:

```python
# Load data for a specific subject
X, Y = extract_data_from_subject(root_dir, N_S, datatype)
```
- `extract_data_from_subject`: Loads EEG recordings for a specific subject
- `root_dir`: Directory where dataset is stored
- `N_S`: Subject number (1-10)
- `datatype`: The type of data to extract ("EEG" in this case)
- `X`: Contains the actual EEG signal data (trials × channels × time samples)
- `Y`: Contains labels indicating condition and direction for each trial

```python
# Select specific time window from the EEG data
X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
```
- `select_time_window`: Crops the EEG data to focus on relevant time period
- `t_start`: Beginning of window in seconds (typically 1.5s)
- `t_end`: End of window in seconds (typically 3.5s)
- `fs`: Sampling frequency (256 Hz)
- This focuses on the portion of the signal containing the mental command activity

```python
# Filter data by specific conditions and classes
X, Y = transform_for_classificator(X, Y, Classes, Conditions)
```
- `transform_for_classificator`: Selects only the trials matching specified conditions and classes
- `Classes`: Selected direction commands (Up, Down, Left, Right)
- `Conditions`: Selected speech type (Inner, Pronounced, Visualized)

### Feature Extraction
After loading and preprocessing, the code extracts frequency-based features:

```python
# Extract frequency band power features from EEG data
features_X = extract_band_power_features(X, fs, bands=bands_dict)
```
- `extract_band_power_features`: Calculates power in different frequency bands
- For each trial and channel, it:
  1. Computes the power spectral density using Welch's method
  2. Averages power in specific frequency ranges (e.g., Alpha: 8-12 Hz, Beta: 13-30 Hz)
  3. Returns these band powers as features for classification

Inside this function:
```python
psds, freqs = mne.time_frequency.psd_array_welch(channel_data, sfreq=sfreq, verbose=False)
```
- Uses Welch's method to calculate the power spectral density
- `psds`: Power values across frequencies
- `freqs`: Frequency bins

```python
band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
band_power = psds[band_indices].mean()
```
- Selects frequencies within the specified band (e.g., Alpha: 8-12 Hz)
- Calculates the average power within that band

### Classification Process
The code then applies machine learning to classify mental commands:

```python
# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features_X, Y, test_size=0.2, random_state=42, stratify=Y
)
```
- `train_test_split`: Divides data into training (80%) and test (20%) sets
- `stratify=Y`: Ensures balanced classes in both sets
- `random_state=42`: Sets a seed for reproducibility

```python
# Train a Linear Discriminant Analysis classifier
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, y_train)
```
- `LinearDiscriminantAnalysis`: A classifier that works well for EEG data
- `classifier.fit`: Trains the model on the training data
- LDA is particularly effective for BCI applications because:
  1. It works well with limited training data
  2. It's computationally efficient
  3. It's less prone to overfitting compared to more complex models

```python
# Evaluate the classifier
y_val_pred = classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
```
- `classifier.predict`: Makes predictions on validation data
- `accuracy_score`: Calculates the proportion of correct predictions
- The classification report also shows precision, recall, and F1-score

### Interactive Widgets
The notebook uses interactive widgets to allow parameter adjustment:

```python
# Subject selection
subject_widget = IntSlider(min=1, max=10, step=1, value=1, description="Subject Number")
```
- Creates a slider to select subjects 1 through 10

```python
# Condition selection (type of speech)
condition_widget = Dropdown(
    options=["Inner", "Pronounced", "Visualized"],
    value="Inner",
    description="Condition",
)
```
- Dropdown menu to select speech condition:
  - "Inner": Mental speech without vocalization
  - "Pronounced": Actually spoken words
  - "Visualized": Visual imagination of words

```python
# Class selection (direction commands)
class_widget = SelectMultiple(
    options=["Up", "Down", "Left", "Right"], value=["Up", "Down"], description="Classes"
)
```
- Multiple selection box for direction commands
- Default selects "Up" and "Down" for binary classification

```python
# Time window selection
time_start_widget = widgets.FloatSlider(min=0, max=4, step=0.1, value=1.5, description="Time Start (s)")
time_end_widget = widgets.FloatSlider(min=0, max=4.5, step=0.1, value=3.5, description="Time End (s)")
```
- Sliders to select the portion of each trial to analyze
- Default values (1.5s to 3.5s) typically capture the mental activity after cue presentation

```python
# Frequency band selection
bands_widget = Dropdown(
    options=band_options,
    value=band_options["Alpha + Beta"],
    description="Frequency Bands",
)
```
- Dropdown to select which brain wave bands to use as features
- Default uses Alpha (8-12 Hz) and Beta (13-30 Hz) bands, which are associated with cognitive tasks

## Understanding Brain Wave Frequency Bands

Each frequency band provides different information about brain activity:

- **Delta (1-3 Hz)**: High amplitude waves most prominent during deep sleep
  - In BCI: May indicate relaxation but generally contains less task-specific information

- **Theta (4-7 Hz)**: Associated with drowsiness, meditation, and creative thinking
  - In BCI: May reflect mental effort or memory encoding/retrieval

- **Alpha (8-12 Hz)**: Dominant during relaxed wakefulness, especially with eyes closed
  - In BCI: Alpha suppression (decrease in power) often indicates active cognitive processing

- **Beta (13-30 Hz)**: Associated with normal waking consciousness and active thinking
  - In BCI: Often increases during motor imagery and cognitive tasks, making it valuable for classification

- **Gamma (30-45 Hz)**: Related to higher cognitive functions and cross-modal sensory processing
  - In BCI: May contain information about complex cognitive processes but also more susceptible to muscle artifacts

## Signal Processing Details

The feature extraction involves several signal processing steps:

1. **Time windowing**: Selecting a portion of the EEG signal where the mental activity is most pronounced
   - Earlier time points (near 0s) contain sensory perception of the cue
   - Later time points (1.5-3.5s) typically contain the mental response

2. **Power spectral density calculation**:
   - Welch's method divides the signal into overlapping segments
   - Computes FFT on each segment
   - Averages the resulting periodograms to reduce noise

3. **Band power extraction**:
   - Averages power values within specific frequency ranges
   - Creates a feature vector that represents the "fingerprint" of brain activity for each mental state

## Detailed Tutorial Instructions

### Getting Started:

1. **Run the initialization cells**: The first cells import necessary libraries and set up the environment

2. **Understanding the data structure**:
   - Each trial represents a single instance of a participant thinking about a direction
   - EEG data is organized as 3D arrays: [trials × channels × time samples]
   - Labels contain information about the condition (speech type) and class (direction)

3. **Experiment with parameters**:
   - Try different subjects to see individual differences in brain patterns
   - Compare different conditions (Inner vs. Pronounced vs. Visualized)
   - Test different frequency bands to see which are most informative
   - Adjust the time window to focus on different parts of neural processing

4. **Interpreting results**:
   - Classification accuracy above 50% for binary classification indicates successful detection
   - The classification report shows detailed performance metrics for each class
   - Higher accuracy suggests stronger neural patterns associated with the mental states

5. **Formulating hypotheses**:
   - Which conditions are easiest to distinguish?
   - Do some subjects show clearer patterns than others?
   - Which frequency bands contain the most task-relevant information?
   - How does the timing of neural activity relate to mental processing?

By working through this tutorial, you'll gain hands-on experience with EEG data analysis, machine learning, and brain-computer interface principles - all accessible through an interactive interface designed for beginners.
