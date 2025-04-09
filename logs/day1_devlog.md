# Inner Speech Classification Model: Development Log

## Current Status: ~50% Accuracy

The existing model classifies "Up" vs "Down" inner speech with approximately 50% accuracy, which is essentially chance level. This suggests the model isn't effectively capturing the neural patterns that differentiate between these two mental states.

## Enhancement Strategy

As both a neuroscientist and ML expert, I'll implement several improvements to boost classification accuracy:

### From a Neuroscience Perspective:

1.  **Multi-subject approach**: Brain patterns vary between individuals, but share common characteristics across people performing the same mental task.
2.  **Frequency-based features**: Mental imagery activates specific frequency bands in the brain:

   *   Alpha (8-12 Hz): Reflects inhibition and is modulated during mental imagery.
   *   Beta (12-30 Hz): Associated with active thinking and motor planning.
   *   Gamma (>30 Hz): Linked to higher cognitive processing.
3.  **Spatial filtering**: Focus on regions where motor imagery and inner speech are processed:

   *   Motor cortex: For directional thinking (central electrodes).
   *   Language areas: For inner speech (typically left-lateralized).
4.  **Time window optimization**: The most discriminative neural activity might occur in specific time windows after cue presentation.

### From an ML Perspective:

1.  **Advanced feature extraction**: Replace simple flattening with better techniques.
2.  **Dimensionality reduction**: Apply techniques like PCA or feature selection to avoid overfitting.
3.  **Cross-validation**: Implement proper CV to ensure model generalizability.
4.  **Hyperparameter tuning**: Find optimal classifier settings.
5.  **Ensemble methods**: Combine multiple classifiers for robust predictions.

## Implementation Plan

1.  Data preprocessing improvements (Multi-subject aggregation, Time windowing)
2.  Feature extraction enhancements (Frequency bands, CSP)
3.  Model selection and optimization (Feature selection, Hyperparameter tuning, Ensemble)
4.  Evaluation and analysis (Accuracy, Confusion Matrix, Comparison)

## Implementation Progress

### Phase 1: Multi-Subject Approach

**Neuroscience rationale**: While brain patterns vary between individuals, certain neural signatures of inner speech should be consistent across subjects. By including data from multiple subjects, we increase our sample size and focus the model on learning the common patterns that distinguish thinking "up" from thinking "down".

**Implementation details**:

*   Using `extract_data_multisubject()` to combine data from multiple participants.
*   Including subjects 1-10 to capture a diverse range of neural responses.
*   Analyzing how individual differences affect classification accuracy.

**Results**: Successfully combined data from all 10 subjects, resulting in 1118 trials (559 "Up" trials and 559 "Down" trials). This balanced dataset provides a solid foundation for our classification task.

**Visualization**: Average EEG signal at a central electrode across all subjects for "Up" vs "Down" thoughts within the selected time window (1.5s-3.5s). Shaded areas represent standard deviation. While subtle, differences might exist, motivating more advanced feature extraction.

![Average EEG Signal Comparison](figures/avg_signal_comparison.png)

### Phase 2: Advanced Feature Extraction

**Neuroscience rationale**: Raw EEG signals contain information across multiple frequency bands, but mental imagery tasks are particularly associated with changes in specific bands:

*   Alpha (8-12 Hz): Suppression indicates active cortical processing.
*   Beta (12-30 Hz): Modulated during motor imagery and cognitive tasks.
*   Gamma (30-100 Hz): Associated with higher-order cognitive processing.

**Implementation details**:

1.  **Frequency-domain features**: Extract power in relevant frequency bands (Theta, Alpha, Beta, Gamma) using `scipy.signal.welch`.
2.  **Common Spatial Patterns (CSP)**: Use `mne.decoding.CSP` to create spatial filters that maximize variance between the "Up" and "Down" classes. This helps isolate the brain regions most involved in differentiating the two mental states.

**Visualization**: Example CSP patterns showing the spatial filters learned. Red areas indicate positive weights, blue areas negative weights. These patterns highlight electrode combinations that are most discriminative between the two classes.

![CSP Patterns](figures/csp_patterns.png)

**Challenges encountered**: Initial implementation had a bug in the frequency extraction function:

*   Error message: `AttributeError: 'numpy.ndarray' object has no attribute 'welch'`
*   Solution: Needed to correctly reference the `scipy.signal.welch` function rather than trying to call it as a method on a NumPy array.

### Phase 3: Advanced Classification Pipeline

**ML rationale**: A more sophisticated pipeline can better handle the high-dimensional, noisy nature of EEG data:

**Implementation details**:

1.  **Feature combination**: Concatenate frequency band power features and CSP features.
2.  **Feature selection**: Use `SelectKBest` with `f_classif` to select the top 50 most discriminative features, reducing dimensionality and potential overfitting.
3.  **Cross-validation**: Implement 5-fold CV (`KFold`) for reliable performance estimation during hyperparameter tuning.
4.  **Hyperparameter tuning**: Use `GridSearchCV` to find optimal parameters for SVM, Random Forest, and LDA classifiers.
5.  **Ensemble methods**: Combine the tuned classifiers using a `VotingClassifier` (soft voting) for improved robustness and accuracy.

Let's track our progress as we implement these changes.

### Analysis Results - 2025-04-05 19:43:24

**Parameters:**

*   Subjects: \[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
*   Data Type: EEG
*   Time Window: 1.5s to 3.5s
*   Group 1: Up direction in Inner condition
*   Group 2: Down direction in Inner condition
*   Model: ensemble
*   **Accuracy: 50.30%**
*   Baseline for Up vs Down in Inner Speech: ~50%
*   Improvement: 0.30%


### Analysis Results - 2025-04-05 19:43:33

**Parameters:**

*   Subjects: \[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
*   Data Type: EEG
*   Time Window: 1.5s to 3.5s
*   Group 1: Up direction in Inner condition
*   Group 2: Down direction in Inner condition
*   Model: ensemble
*   **Accuracy: 50.30%**
*   Baseline for Up vs Down in Inner Speech: ~50%
*   Improvement: 0.30%


### Analysis Results - 2025-04-05 19:44:15

**Parameters:**

*   Subjects: \[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
*   Data Type: EEG
*   Time Window: 1.5s to 3.5s
*   Group 1: Up direction in Pronounced condition
*   Group 2: Down direction in Inner condition
*   Model: ensemble
*   **Accuracy: 89.72%**


### Analysis Results - 2025-04-05 19:53:12

**Parameters:**

*   Subjects: \[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
*   Data Type: EEG
*   Time Window: 1.5s to 3.5s
*   Group 1: Up direction in Inner condition
*   Group 2: Up direction in Pronounced condition
*   Model: ensemble
*   **Accuracy: 81.82%**


## Detailed Implementation Summary (Based on day1.py - 2025-04-07)

Following the enhancement strategy, `day1.py` implements a significantly improved pipeline compared to the baseline. Here's a breakdown aligned with our Neuroscience and ML perspectives, with added explanations for clarity:

**1. Multi-Subject Data Aggregation (Neuroscience & ML)**

*   **Implementation**: Data from subjects 1-10 (`subject_list = list(range(1, 11))`) were loaded using `extract_data_from_subject` and combined. Caching was implemented to speed up reloading.
    *   **What**: Loads EEG data for each specified subject, filters it for the desired conditions ("Inner") and classes ("Up", "Down"), and stacks the resulting trials together into single large arrays (`X`, `Y`).
    *   **Jargon**:
        *   `extract_data_from_subject`: Custom function to load preprocessed EEG data (`.fif` files) for a single subject.
        *   `np.vstack`: NumPy function to stack arrays vertically (row-wise), used for the `X` data (trials).
        *   `np.concatenate`: NumPy function to join arrays along an existing axis, used for the `Y` labels.
        *   Caching (`load_from_cache`, `save_to_cache`): Saves the results of computationally expensive steps (like loading/processing data) to disk (`.pkl` files in `cache/`). If the script is run again with the same parameters, it loads the cached result instead of recomputing, saving significant time. Keys are generated using `hashlib.md5` based on relevant parameters.
    *   **Why**: Combining data across subjects aims to build a model that learns the general neural patterns associated with "Up" vs "Down" inner speech, rather than patterns specific to one individual. Caching makes iterative development much faster.
*   **Neuro-Rationale**: Leverages common neural signatures across individuals for inner speech tasks ("Up" vs "Down").
*   **ML-Rationale**: Increases dataset size and diversity, leading to a more generalizable model. Addresses Phase 1.
*   **Visualization**: The average signal plot (shown in Phase 1 above) illustrates the combined data used.

**2. Focused Time Window (Neuroscience)**

*   **Implementation**: Data was filtered to the 1.5s - 3.5s window post-cue using `select_time_window`.
    *   **What**: Selects only the portion of each trial's EEG data that falls between 1.5 and 3.5 seconds after the instruction cue was presented.
    *   **Jargon**:
        *   `select_time_window`: Custom function that slices the time dimension of the EEG data array based on start/end times and the sampling frequency (`fs`).
    *   **Why**: The experimental protocol suggests this time window is when the participant is actively performing the inner speech task. Focusing on this window reduces noise and irrelevant data from before or after the task execution. Aligns with Strategy Point 4 (Neuro).
*   **Neuro-Rationale**: Concentrates analysis on the time period most relevant to the execution of the mental task.

**3. Advanced Feature Engineering (Neuroscience & ML)**

*   **Implementation**: Two primary feature types were extracted and combined (`np.hstack`):
    *   **Frequency Band Power**: Calculated using `extract_frequency_features` (employing `scipy.signal.welch`) for Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz), and Gamma (30-100Hz) bands. Caching applied.
        *   **What**: For each trial and EEG channel, calculates the average power within specific frequency ranges (bands).
        *   **Jargon**:
            *   `extract_frequency_features`: Custom function implementing the power calculation.
            *   `scipy.signal.welch`: A standard method to estimate the power spectral density (PSD) of a signal. It divides the signal into segments, calculates the spectrum for each, and averages them, reducing noise.
            *   Frequency Bands (Theta, Alpha, Beta, Gamma): Standard ranges of brainwave frequencies associated with different cognitive states (see Neuro-Rationale).
            *   Power Spectral Density (PSD): A measure of the signal's power content versus frequency.
        *   **Why**: Different mental states often manifest as changes in the power of specific brainwave frequencies. Extracting these powers provides features potentially more informative than the raw signal. (Strategy Point 2, Neuro).
    *   **Common Spatial Patterns (CSP)**: Implemented using `mne.decoding.CSP` (`n_components=6`) to find optimal spatial filters. CSP patterns were visualized (`mne.viz.plot_topomap`). Caching applied.
        *   **What**: Finds linear combinations of EEG channels (spatial filters) that maximize the variance (signal power) for one class ("Up") while minimizing it for the other class ("Down"), and vice-versa. The transformed data represents the signal projected onto these filters.
        *   **Jargon**:
            *   `mne.decoding.CSP`: MNE-Python's implementation of the CSP algorithm.
            *   `n_components=6`: Specifies how many pairs of spatial filters to compute. CSP finds filters in pairs; the first filter maximizes variance for class A, the second for class B, etc. 6 components mean 3 pairs.
            *   Spatial Filter: A set of weights applied to each EEG channel. Summing the weighted channel signals creates a new "virtual" channel sensitive to specific spatial patterns.
            *   `csp.fit_transform`: Computes the CSP filters based on the training data (`X`, `Y`) and applies them to `X`.
            *   `csp.filters_`: The actual spatial filters (weights for each channel). Used for plotting topomaps.
            *   `mne.viz.plot_topomap`: MNE function to visualize data distributed over the scalp according to electrode positions.
        *   **Why**: CSP is highly effective for EEG classification, especially motor imagery, as it directly targets spatial patterns that differentiate between the classes. It acts as both a dimensionality reduction and feature extraction technique. (Strategy Point 3, Neuro; Strategy Point 1, ML).
    *   **Feature Combination**: `np.hstack` was used to combine frequency and CSP features horizontally into a single feature vector per trial.
*   **Neuro-Rationale**: Extracts power from relevant frequency bands and identifies discriminative spatial patterns.
*   **ML-Rationale**: Creates a richer feature set by combining frequency and spatial information. Addresses Phase 2.
*   **Visualization**: The CSP patterns plot (shown in Phase 2 above) visualizes the spatial filters derived.

**4. Feature Selection (ML)**

*   **Implementation**: `SelectKBest` with `f_classif` was used to select the top `k=50` features from the combined set (`X_combined`). Caching applied.
    *   **What**: Reduces the number of features (columns in the data matrix) by selecting only the `k` features that have the strongest statistical relationship with the class labels (`Y`).
    *   **Jargon**:
        *   `SelectKBest`: Scikit-learn class for selecting features based on univariate statistical tests.
        *   `f_classif`: The statistical test used (ANOVA F-value). It assesses if the means of the feature values are significantly different across the classes.
        *   `k=50`: The desired number of top features to keep.
    *   **Why**: Reduces dimensionality, which can prevent overfitting (where the model learns noise specific to the training data) and potentially improve model performance and training speed by focusing on the most relevant information. (Strategy Point 2, ML). Addresses Phase 3.
*   **ML-Rationale**: Reduces dimensionality, prevents overfitting, and focuses the model on the most discriminative information.

**5. Sophisticated Classification Pipeline (ML)**

*   **Implementation**:
    *   **Train/Test Split**: Data split using `train_test_split` (30% test set, `random_state=42` for reproducibility).
        *   **What**: Divides the selected features (`X_selected`) and labels (`Y`) into two sets: one for training the models (70%) and one for evaluating their performance on unseen data (30%).
        *   **Why**: Essential for evaluating how well a model generalizes to new data it hasn't been trained on.
    *   **Hyperparameter Tuning**: `GridSearchCV` with 5-fold cross-validation (`KFold`) was used to find optimal parameters for SVM (`SVC`), Random Forest (`RandomForestClassifier`), and LDA (`LinearDiscriminantAnalysis`).
        *   **What**: Systematically searches through predefined combinations of parameters (hyperparameters) for each classifier type to find the combination that yields the best performance based on cross-validation within the *training* set.
        *   **Jargon**:
            *   `GridSearchCV`: Scikit-learn tool for exhaustive hyperparameter search.
            *   `KFold(n_splits=5)`: Cross-validation strategy. Splits the training data into 5 folds; trains on 4, validates on 1, repeated 5 times so each fold is used for validation once. Performance is averaged across folds.
            *   Hyperparameters: Parameters set *before* training (e.g., `C` in SVM, `n_estimators` in Random Forest).
            *   `SVC`, `RandomForestClassifier`, `LinearDiscriminantAnalysis`: Different types of classification algorithms.
        *   **Why**: Optimizes the performance of each individual classifier type. Cross-validation provides a more reliable estimate of performance than a single train/validation split. (Strategy Point 3 & 4, ML).
    *   **Ensemble Model**: A `VotingClassifier` (soft voting) combined the *best* tuned versions of SVM, RF, and LDA models.
        *   **What**: Creates a meta-classifier that combines the predictions of the individual tuned models. 'Soft' voting averages the predicted probabilities from each model and predicts the class with the highest average probability.
        *   **Jargon**:
            *   `VotingClassifier`: Scikit-learn tool for combining multiple models.
            *   Soft Voting: Averages probabilities. (Hard voting would take a majority vote based on predicted classes).
        *   **Why**: Ensembles often achieve better and more robust performance than any single constituent model by leveraging their diverse strengths and reducing the impact of individual model weaknesses. (Strategy Point 5, ML).
    *   **Caching**: Trained models (`GridSearchCV` objects and the ensemble) were cached.
*   **ML-Rationale**: Employs robust evaluation (CV), optimizes individual models (Tuning), and leverages multiple models (Ensemble) for improved accuracy and robustness. Addresses Phase 3.

**6. Evaluation and Comparison**

*   **Implementation**: Accuracy was calculated for baseline, individual tuned models, and the ensemble model on the *test set*. A confusion matrix (`seaborn.heatmap`) and classification report (`sklearn.metrics.classification_report`) were generated for the ensemble model. A bar chart (`matplotlib.pyplot.bar`) compares the performance of all models against baseline and chance level.
    *   **What**: Assesses the performance of the final trained models on the held-out test data.
    *   **Jargon**:
        *   Accuracy: The proportion of correctly classified trials.
        *   Confusion Matrix: A table showing correct and incorrect predictions for each class (True Positives, True Negatives, False Positives, False Negatives).
        *   Classification Report: Provides precision, recall, and F1-score for each class.
            *   Precision: Of the trials predicted as class X, how many were actually class X? (TP / (TP + FP))
            *   Recall: Of the actual class X trials, how many were correctly predicted? (TP / (TP + FN))
            *   F1-score: Harmonic mean of precision and recall.
        *   Baseline Model: A simple SVM model trained on flattened raw data, used as a reference point.
    *   **Why**: Quantifies the performance of the developed pipeline and compares it against a simpler approach and chance level to demonstrate the effectiveness of the implemented improvements.
*   **Rationale**: Provides quantitative assessment and detailed performance analysis.
*   **Visualizations**:
    *   **Confusion Matrix**: Visualizes the ensemble model's predictions vs true labels on the test set. Helps identify if the model confuses one class for another. (See #attachment_image_Ensemble_Model_Confusion_Matrix.png)

        ![Ensemble Confusion Matrix](figures/confusion_matrix.png)

    *   **Model Comparison**: Bar chart comparing test accuracies of all models. (See #attachment_image_Classification_Accuracy_Comparison.png)

        ![Model Accuracy Comparison](figures/model_comparison.png)

**Outcome**: A slight accuracy improvement over the baseline for the challenging "Inner Up" vs "Inner Down" task (reaching 50.3% with the ensemble). While the improvement is marginal for this specific comparison, the framework (multi-subject, advanced features, tuning, ensemble) is established and demonstrated much higher accuracy on easier comparisons (e.g., Pronounced vs Inner, achieving >80% in other runs). Caching dramatically speeds up iterative development. The low accuracy for Inner vs Inner suggests these specific mental states are very difficult to distinguish with the current features and models.
