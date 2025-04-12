
# Project Report: Decoding Inner Speech from EEG

**TLDR:** This project analyzes EEG data from 10 participants performing cued mental tasks (inner speech, pronounced speech, visualization of directions) using the open-access Nieto et al. (2022) dataset (ds003626). The goal is to decode these mental states, particularly directional inner speech ("Up", "Down", etc.). Neuroscience principles guide the analysis: inner speech likely involves motor/language brain areas, manifesting as changes in EEG oscillations (brainwaves) and spatial patterns. Preprocessing cleans the EEG signal (filtering, artifact removal via ICA, EMG control). Analysis focuses on a specific time window (1.5-3.5s post-cue) where the task is active. Features are extracted based on neuroscience: frequency band power (Theta, Alpha, Beta, Gamma) captures oscillatory changes, and Common Spatial Patterns (CSP) identifies discriminative spatial activity patterns across the scalp. Data from all subjects are combined to find generalizable neural signatures. Visualizations like ERPs and topomaps help interpret the neural activity.

## Neuroscience Foundation and Data Acquisition

This project aims to classify different types of inner speech (specifically directional thoughts like "Up", "Down", "Left", "Right") and related mental states using Electroencephalography (EEG) data. The approach is grounded in understanding how these cognitive processes manifest in brain activity.

**1. Data Source:**
   - We utilized the open-access "Inner Speech Dataset" (ds003626) published by Nieto et al. (2022) on OpenNeuro (#`Inner_Speech_Dataset/README.md`). This dataset is crucial as it provides high-density EEG recordings specifically designed to investigate inner speech and related cognitive states.
   - The dataset contains EEG recordings from 10 healthy participants. Data was acquired using a 128-channel Biosemi ActiveTwo system, initially sampled at 2048 Hz. For analysis, it was typically downsampled (e.g., to 256 Hz in #`day1.py`, #`day1_ui.py`) to reduce computational load while retaining relevant frequencies.

**2. Experimental Paradigm:**
   - The experiment involved participants responding to visual cues (arrows, #`Set_up_Estimulos.m`). Based on the cue and instructions for a given block, they performed one of three tasks (Conditions):
      - **Inner Speech:** Silently pronounce the direction indicated by the cue (e.g., think "Up" for an upward arrow).
      - **Pronounced Speech:** Overtly speak the direction.
      - **Visualized:** Mentally visualize the shape of the arrow cue.
   - This design allows for comparing brain activity during inner speech to both overt speech (sharing production pathways) and a non-linguistic visualization task (control).
   - Precise timing is critical. Event markers (triggers sent via LSL, #`send_value_pp.m`, #`Set_up_Marcadores.m`) were embedded in the EEG data to synchronize the recordings with task events (cue onset, response period, etc.). These markers (e.g., 31='Arriba'/Up, 32='Abajo'/Down, 21='Pronunciada', 22='Imaginada'/Inner, 23='Visualizacion') are essential for segmenting the data into meaningful trials (#`InnerSpeech_preprocessing.py`, #`Events_analysis.py`).

**3. Neural Basis of Inner Speech:**
   - Inner speech, or "talking in your head," is a complex cognitive function. Neuroscientific hypotheses suggest it engages a network of brain regions that partially overlaps with those used for overt speech production and comprehension.
   - Key areas potentially involved include:
      - **Motor and Premotor Cortex:** Especially supplementary motor area (SMA), involved in planning speech articulation, even if not executed overtly. Differences in directional inner speech ("Up" vs. "Down") might manifest as distinct preparatory motor patterns (#`Archive/Tutorial_otherClassifiers.md`).
      - **Broca's and Wernicke's Areas:** Classical language areas (often left-lateralized) involved in speech production and comprehension, respectively. Their activation during inner speech is debated but plausible.
      - **Auditory Cortex:** May be involved in the "hearing" aspect of inner speech.
   - The EEG signals associated with inner speech are subtle compared to overt actions. Decoding relies on identifying faint but consistent patterns of neural synchrony/desynchrony (oscillations) and spatial activation across the scalp.

**4. EEG Signal Processing (Neuroscience Focus):**
   - Raw EEG data is noisy. Preprocessing steps, guided by neurophysiological understanding, are vital to enhance the task-related signal (#`InnerSpeech_preprocessing.py`).
      - *Referencing:* Data was re-referenced (e.g., to linked mastoids using channels EXG1, EXG2) to remove noise common to all electrodes, such as electrical noise or broad physiological signals, improving the signal-to-noise ratio for localized brain activity.
      - *Filtering:*
         - Notch filtering (e.g., 50 Hz) removes specific frequencies associated with power line interference, which is not physiological.
         - Band-pass filtering (e.g., 0.5-100 Hz or 1-100 Hz) retains frequencies where most relevant brain activity occurs (delta, theta, alpha, beta, gamma waves) while removing slow drifts and very high-frequency noise.
      - *Epoching:* Continuous EEG data was segmented into trials (epochs) time-locked to the stimulus cue (e.g., from -0.5s before the cue to 4s after). This allows averaging across trials or analyzing specific time windows relative to the task event.
      - *Artifact Handling:* Non-brain electrical signals (artifacts) can contaminate EEG.
         - **ICA (Independent Component Analysis):** A powerful technique used to separate statistically independent sources mixed in the EEG signals. Components strongly correlated with eye movements (recorded by EOG channels EXG3-EXG6) or blinks (often captured by frontal EEG channels or EXG5/EXG6) were identified and removed (#`InnerSpeech_preprocessing.py`). This prevents eye movements from being misinterpreted as brain activity related to the task.
         - **EMG Control:** Muscle activity, especially from the mouth/jaw during inner or pronounced speech, can create high-frequency noise. An additional check (#`EMG_Control.py`) specifically targeted trials potentially contaminated by subtle mouth movements by comparing the signal power in designated EMG channels (EXG7, EXG8) during the task period to a baseline period. Trials exceeding a threshold were flagged or potentially removed (#`EMG_control_single_th`).
      - *Montage:* Applying a standard electrode layout (Biosemi 128-channel, #`InnerSpeech_preprocessing.py`, #`PSD_representation.py`) is necessary for accurate topographical visualization (mapping activity across the scalp) and for spatial filtering techniques like CSP.

   - **Time Window Selection (#`day1.py`, #`day2.py`, #`logs/day1_devlog.md`):**
      - The cognitive processes involved in the task unfold over time. Analysis was often focused on a specific window *after* the cue (e.g., 1.5s to 3.5s).
      - *Rationale:* This window is assumed to capture the core period when the participant is actively performing the mental task (e.g., formulating and executing the inner speech direction), excluding the initial cue perception/processing phase and post-task activity. This aims to maximize the signal related to the specific cognitive state of interest and reduce noise from other periods (#`logs/day1_devlog.md`).

**5. Feature Engineering (Neuroscience Rationale):**
   - Raw EEG time-series data is high-dimensional. Feature engineering extracts specific characteristics thought to reflect the underlying neural processes differentiating the mental states.
   - **Frequency Domain Features (#`day1.py`, #`extract_frequency_features`, #`Archive/Tutorial_otherClassifiers.md`, #`logs/day1_devlog.md`):**
      - Brain activity is oscillatory. Different mental states are associated with changes in the power (amplitude) of oscillations within specific frequency bands.
      - Power Spectral Density (PSD) was calculated (using Welch's method, `scipy.signal.welch` in #`day1.py`, `mne.time_frequency.psd_array_welch` in #`Archive/main.ipynb`) for each channel within standard frequency bands:
         - **Theta (4-8Hz):** Often linked to memory processes, attention.
         - **Alpha (8-12Hz):** Typically associated with relaxation or cortical inhibition. *Suppression* (decrease in power) often indicates active processing in the underlying area.
         - **Beta (12-30Hz):** Linked to active thinking, concentration, and motor planning/execution (including speech). Changes in beta power over motor areas are expected during directional inner speech.
         - **Gamma (30-100Hz):** Associated with feature binding, complex cognition, and potentially heightened neural communication.
      - *Rationale:* The hypothesis is that the *pattern* of power changes across these bands and channels differs between, for example, thinking "Up" versus thinking "Down" (#`logs/day1_devlog.md`, #`Archive/Tutorial_otherClassifiers.md`).
   - **Spatial Domain Features (CSP) (#`day1.py`, #`mne.decoding.CSP`, #`Archive/Tutorial_otherClassifiers.md`, #`logs/day1_devlog.md`):**
      - Common Spatial Patterns (CSP) is a supervised spatial filtering technique highly effective for classifying EEG states, particularly motor imagery.
      - It finds linear combinations of electrode signals (spatial filters) that maximize the variance (signal power) for one mental state (e.g., "Inner Up") while simultaneously minimizing it for another (e.g., "Inner Down").
      - *Rationale:* CSP directly targets the *spatial distribution* of brain activity that best distinguishes the classes. The resulting CSP components are "virtual channels" designed to capture the most discriminative neural sources. For directional inner speech, these patterns might highlight differential activation in motor planning areas or related networks (#`Archive/Tutorial_otherClassifiers.md`, #`logs/day1_devlog.md`). Visualizing the CSP filters as topomaps (#`day1.py`) can offer insights into which scalp regions contribute most to the classification.

**6. Multi-Subject Analysis (#`day1.py`, #`Data_extractions.py`, #`logs/day1_devlog.md`):**
   - Data from multiple subjects (often all 10, #`subject_list = list(range(1, 11))` in #`day1.py`) were aggregated (#`extract_data_from_subject`, #`np.vstack`, #`np.concatenate`).
   - *Rationale:* Individual brain anatomy and function vary, leading to differences in EEG patterns. However, the core neural processes underlying a specific task (like inner speech) are expected to share common features across individuals. Combining data allows machine learning models to learn these *generalizable* signatures, reducing the influence of subject-specific idiosyncrasies and improving the potential for a BCI that works across users (#`logs/day1_devlog.md`).

**7. Visualization for Neuroscientific Insight:**
   - Visualizing processed data helps interpret the neural correlates of the tasks.
   - **ERPs (Event-Related Potentials) (#`Plot_ERPs.py`):** Averaging the time-locked epochs for a specific condition/class reveals the consistent voltage fluctuations related to the event. While less common for oscillatory analysis, ERPs can show processing stages.
   - **Topomaps (#`Plot_TRF_topomaps.py`, #`day1.py`):** Plotting the spatial distribution of EEG power in a frequency band, or the weights of CSP filters, across the scalp. This helps infer the likely brain regions involved (e.g., activity focused over central areas might suggest motor cortex involvement).
   - **PSD Plots (#`PSD_representation.py`, #`PSD_plot_PSD.py`):** Graphing power versus frequency for specific channels or averaged across channels/trials. This allows direct comparison of oscillatory activity between conditions (e.g., comparing the Beta band power during "Inner Up" vs. "Inner Down").


## Machine Learning Pipeline for Classification

Once the EEG data was preprocessed and relevant features were extracted based on neuroscientific principles, a machine learning pipeline was employed to classify the different mental states. The goal was to train models that could accurately predict the participant's mental task (e.g., distinguishing "Inner Up" from "Inner Down") based on the engineered features.

**1. Feature Set:**
   - The input to the machine learning models consisted of the features derived from the neuroscience-driven processing (#`day1.py`, #`day2_analysis_pipeline.py`):
      - **Frequency Band Power:** Average power in Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz), and Gamma (30-100Hz) bands for each channel (#`extract_frequency_features`).
      - **Common Spatial Patterns (CSP):** Features derived from applying CSP filters (typically 6 components, meaning 3 pairs of filters) designed to maximize the variance difference between the two classes being compared (#`mne.decoding.CSP`).
   - These two sets of features were concatenated (`np.hstack`) to create a combined feature vector for each trial, aiming to capture both spectral and spatial discriminative information.

**2. Dimensionality Reduction / Feature Selection:**
   - The combined feature set could be high-dimensional (Channels * Bands + CSP Components). To mitigate the risk of overfitting (where the model learns noise specific to the training data) and improve computational efficiency, feature selection was applied (#`day1.py`, #`day2_analysis_pipeline.py`).
   - **Method:** `SelectKBest` from scikit-learn was used with the `f_classif` scoring function (ANOVA F-value). This selects the `k` features (typically `k=50` in #`day2_analysis_pipeline.py`) that show the strongest statistical relationship (difference in means) between the classes being classified.
   - *Rationale:* This step focuses the model on the most informative features, potentially improving generalization to unseen data.

**3. Data Splitting and Validation Strategy:**
   - To train and evaluate the models reliably, the data was split:
      - **Train/Test Split:** The dataset (after feature selection) was divided into a training set (e.g., 70% of trials) and a testing set (e.g., 30%) using `train_test_split` (#`day1.py`, #`day2_analysis_pipeline.py`). The test set was held out and used only for the final evaluation of the trained models. Stratification (`stratify=Y`) was often used to ensure similar class proportions in both train and test sets, which is important for imbalanced datasets.
      - **Cross-Validation (within Training):** During hyperparameter tuning (see below), K-Fold Cross-Validation (typically 5-fold, `KFold(n_splits=5)`) was applied *within the training set*. This involves splitting the training data into `k` folds, training on `k-1` folds, and validating on the remaining fold, rotating the validation fold. Performance is averaged across the `k` iterations.
   - *Rationale:* The train/test split provides an unbiased estimate of generalization performance. Cross-validation within the training set allows for robust hyperparameter selection without "peeking" at the final test set.

**4. Classifier Selection and Hyperparameter Tuning:**
   - Several standard machine learning classifiers suitable for BCI applications were evaluated (#`day1.py`, #`day2_analysis_pipeline.py`):
      - **Support Vector Machine (SVM):** `SVC`. A powerful classifier that finds an optimal hyperplane to separate classes. Effective in high-dimensional spaces. Kernels like 'rbf' and 'linear' were often explored.
      - **Random Forest:** `RandomForestClassifier`. An ensemble method based on multiple decision trees. Robust to overfitting and good at capturing non-linear relationships.
      - **Linear Discriminant Analysis (LDA):** `LinearDiscriminantAnalysis`. A simpler linear classifier often effective for EEG data, especially when data is limited or classes are reasonably separable linearly. It explicitly models class distributions.
   - **Hyperparameter Tuning:** For each classifier type, `GridSearchCV` was used in conjunction with the K-Fold cross-validation strategy on the training data.
      - *What:* `GridSearchCV` systematically trains and evaluates the classifier using different combinations of predefined hyperparameters (e.g., `C` and `gamma` for SVM; `n_estimators` and `max_depth` for Random Forest).
      - *Goal:* To find the hyperparameter settings that yield the best average performance across the cross-validation folds for each classifier type.
   - *Rationale:* Different classifiers have different strengths and weaknesses. Tuning hyperparameters optimizes the performance of each classifier for the specific dataset and feature set.

**5. Ensemble Modeling:**
   - To potentially achieve better and more robust performance than any single classifier, an ensemble model was created using `VotingClassifier` (#`day1.py`, #`day2_analysis_pipeline.py`).
   - **Method:** The *best* versions (with tuned hyperparameters found via `GridSearchCV`) of the individual classifiers (SVM, RF, LDA) were combined. 'Soft' voting was typically used, meaning the ensemble prediction is based on the average of the predicted probabilities from each constituent model, rather than just a majority vote on the predicted classes.
   - *Rationale:* Ensembles often generalize better by averaging out the biases or errors of individual models, leading to more stable and accurate predictions.

**6. Evaluation Metrics:**
   - Model performance was assessed on the held-out test set using several standard metrics (#`day1.py`, #`day2_analysis_pipeline.py`, #`day2.py`):
      - **Accuracy:** Overall percentage of correctly classified trials. The primary metric, but can be misleading if classes are imbalanced.
      - **Precision:** Of the trials predicted as class A, how many were actually class A? (Measures prediction correctness).
      - **Recall (Sensitivity):** Of all the actual class A trials, how many were correctly predicted? (Measures ability to find all instances of a class).
      - **F1-Score:** The harmonic mean of Precision and Recall. Provides a single score balancing both concerns.
      - **Confusion Matrix:** A table showing the counts of correct and incorrect predictions for each class (True Positives, True Negatives, False Positives, False Negatives). Visualized using `seaborn.heatmap` (#`day1.py`).
      - **Classification Report:** Text summary from `sklearn.metrics.classification_report` providing precision, recall, and F1-score per class.
      - **Class Balance Ratio:** Calculated in `day2.py` as `min(counts) / max(counts)` to quantify potential imbalance between the two classes being compared. Reported alongside other metrics in #`day2_pairwise_results.csv`.

**7. Systematic Pairwise Analysis (`day2.py`):**
   - To comprehensively map the discriminability between different mental states, a systematic analysis was performed in #`day2.py`.
   - **Process:**
      - All 12 unique states (3 Conditions x 4 Classes) were defined.
      - All 66 unique pairwise combinations of these states were generated (`itertools.combinations`).
      - The entire enhanced analysis pipeline (data loading, feature extraction, selection, tuned ensemble training, evaluation) defined in #`day2_analysis_pipeline.py` was run for *each* of these 66 pairs.
   - **Output:** Results (accuracy, precision, recall, F1, balance ratio, trial counts, time taken) for each pair were logged progressively to #`logs/day2_devlog.md` and saved in a final summary table #`logs/day2_pairwise_results.csv`.
   - *Rationale:* This provides a detailed overview of which mental state contrasts are easy or difficult to classify with the current approach, highlighting potential candidates for BCI control or areas needing further methodological improvement.

**8. Caching:**
   - To accelerate development and repeated analyses, caching was implemented using `pickle` and `hashlib` (#`day1.py`, #`day2_analysis_pipeline.py`). Intermediate results (loaded data, extracted features, selected features, trained models) were saved to disk. If the script was run again with the same parameters, the cached results were loaded instead of recomputing, saving significant time.

**9. Baseline Comparison:**
   - In #`day1.py`, the performance of the enhanced pipeline (multi-subject, advanced features, tuning, ensemble) was explicitly compared against a simpler baseline model (SVM applied to flattened raw EEG data from the selected time window). This served to quantify the improvement gained from the more sophisticated neuroscience-informed feature engineering and ML techniques.

This machine learning pipeline represents a systematic approach to decoding mental states from EEG, incorporating feature engineering, selection, robust validation, model tuning, and ensemble methods to maximize classification performance. The pairwise analysis provides a broad characterization of the dataset's potential.

## Devlog Entry: `Archive/main.ipynb` - Interactive Exploration Notebook

**Purpose:** This Jupyter notebook (`Archive/main.ipynb`) appears to be an early or tutorial version focused on interactive exploration of the dataset and basic classification for a *single subject* at a time. It utilizes `ipywidgets` to allow users to dynamically change parameters like subject number, condition, classes, time window, and frequency bands, and observe the impact on classification performance using a simple Linear Discriminant Analysis (LDA) model.

**Key Implementation Steps:**

1.  **Imports & Setup:**
    *   Imports standard libraries (`mne`, `numpy`, `sklearn`) and crucially, `ipywidgets` for interactivity.
    *   Imports data processing functions (`extract_data_from_subject`, `select_time_window`, `transform_for_classificator`) from the `Inner_Speech_Dataset/Python_Processing` directory.
    *   Basic setup like setting random seed and MNE log level.

2.  **Feature Extraction (`extract_band_power_features`):**
    *   Defines a function to calculate Power Spectral Density (PSD) using `mne.time_frequency.psd_array_welch` for each channel and trial.
    *   Averages the PSD values within specified frequency bands (e.g., Alpha, Beta) selected via widgets.
    *   *Note:* This uses MNE's Welch implementation, slightly different from the `scipy.signal.welch` used later in `day1.py`.

3.  **Main Analysis Function (`run_analysis`):**
    *   This core function is designed to be driven by the interactive widgets.
    *   Takes parameters: `subject_number`, `selected_condition`, `class_choices`, `time_start`, `time_end`, `selected_bands`, `classifier_choice`.
    *   **Data Loading/Preprocessing:**
        *   Loads data for the selected single subject (`extract_data_from_subject`).
        *   Applies the selected time window (`select_time_window`).
        *   Filters data based on the chosen condition and classes (`transform_for_classificator`). *Note: The widget allows selecting multiple classes, but the code seems structured for binary comparison based on how `Conditions` and `Classes` are formatted before passing to `transform_for_classificator`.*
    *   **Feature Extraction:** Calls `extract_band_power_features` with the selected bands.
    *   **Classification & Evaluation:**
        *   Splits data into training and *validation* sets (`train_test_split`). *Note: No separate final test set is used here, evaluation is on a validation split.*
        *   Trains a Linear Discriminant Analysis (`LinearDiscriminantAnalysis`) classifier (the only option implemented despite the widget).
        *   Predicts on the validation set and calculates accuracy (`accuracy_score`) and a classification report (`classification_report`).
        *   Prints the results to the notebook output.

4.  **Interactivity (`ipywidgets`):**
    *   Defines various widgets (`IntSlider`, `Dropdown`, `SelectMultiple`, `FloatSlider`) to control the parameters for `run_analysis`.
    *   Uses `interact` to link these widgets to the `run_analysis` function, allowing users to change parameters and re-run the analysis dynamically within the notebook.

**Role & Comparison to Later Scripts:**

*   This notebook served as an initial, interactive tool for exploring the data and basic classification concepts for individual subjects.
*   It predates the more systematic, multi-subject, and advanced pipeline developed in `day1.py` and `day2.py`.
*   Key differences from `day1.py`/`day2.py`:
    *   Focus on single-subject, interactive analysis vs. multi-subject batch processing.
    *   Uses MNE's Welch for PSD vs. Scipy's Welch.
    *   Only implements band power features (no CSP).
    *   Only implements LDA classifier.
    *   Uses a train/validation split vs. train/test split with cross-validation for tuning.
    *   Lacks the caching mechanisms implemented later.

This notebook was likely valuable for initial data familiarization and hypothesis testing before developing the more robust analysis scripts.


## Devlog Entry: `src/day1.py` - Building an Improved Classifier

**Date:** 2025-04-11

**Goal:** The primary objective of `day1.py` was to significantly improve upon the baseline (~50% chance level) accuracy for classifying "Inner Up" vs "Inner Down" mental states using EEG data. We aimed to achieve this by systematically incorporating neuroscience insights and more advanced machine learning techniques, moving beyond the basic approach explored in earlier notebooks like `Archive/main.ipynb`.

**Thought Process & Implementation Steps:**

1.  **Setup & Configuration:**
    *   **Imports:** Imported necessary libraries: `mne` for EEG processing, `numpy` for numerical operations, `sklearn` for ML (classifiers, metrics, feature selection, pipelines), `scipy.signal` for feature extraction (Welch's method), `matplotlib`/`seaborn` for plotting, and utilities like `os`, `pickle`, `hashlib`, `time` for file management, caching, and timing.
    *   **Configuration:** Defined key parameters upfront: `root_dir` (dataset location), `datatype` ("EEG"), `fs` (sampling rate, 256 Hz), `t_start`/`t_end` (1.5s to 3.5s post-cue time window), `subject_list` (all 10 subjects), target `Conditions` ("Inner") and `Classes` ("Up", "Down"). Created directories for `figures` and `cache`.
    *   **Rationale:** Centralized configuration makes the script easier to modify and understand. Setting the random seed ensures reproducibility. Suppressing excessive logging (`mne.set_log_level`) keeps the output clean.

2.  **Improvement 1: Multi-Subject Analysis & Caching:**
    *   **Neuro-Rationale:** Individual brain patterns vary, but the core neural signatures for a task should be somewhat consistent across people. Combining data from multiple subjects helps the model learn these generalizable patterns, rather than overfitting to one person's idiosyncrasies.
    *   **ML-Rationale:** Increases the size and diversity of the training dataset, which generally leads to more robust and generalizable models.
    *   **Implementation:**
        *   Iterated through `subject_list` (1-10).
        *   Used `extract_data_from_subject` to load data for each subject.
        *   Applied the selected time window (1.5s-3.5s) using `select_time_window`.
        *   Filtered for the specific `Conditions` ("Inner") and `Classes` ("Up", "Down") using `transform_for_classificator`.
        *   Appended valid data (`X_filtered`, `Y_filtered`) to lists (`X_all_subjects`, `Y_all_subjects`).
        *   Combined lists into final `X` and `Y` arrays using `np.vstack` and `np.concatenate`.
    *   **Caching:** Implemented caching for this step. Generated a unique `cache_key` based on parameters (`subject_list`, `datatype`, time window, conditions, classes) using `create_cache_key`. Used `load_from_cache` to check if processed data already existed; if not, performed the loading/processing and saved the result (`X`, `Y`) using `save_to_cache`.
    *   **Why Cache?** Loading and preprocessing data for 10 subjects is time-consuming. Caching allows us to skip this step on subsequent runs with the same parameters, drastically speeding up development and experimentation.
    *   **Visualization:** Plotted the average EEG signal over time for "Up" vs "Down" trials at a central electrode, averaged across all subjects (#`figures/avg_signal_comparison.png`). This provides a visual check of the combined data.

3.  **Improvement 2: Advanced Feature Extraction (Frequency + CSP):**
    *   **Neuro-Rationale:** Inner speech and motor imagery are known to modulate brain oscillations in specific frequency bands (Alpha, Beta, Gamma) and involve specific spatial patterns of activation (e.g., over motor cortex). We need features that capture both aspects.
    *   **ML-Rationale:** Raw EEG time-series data is very high-dimensional and noisy. Extracting meaningful features improves signal-to-noise ratio and provides the classifier with more informative input.
    *   **Implementation:**
        *   **Frequency Features:** Defined `extract_frequency_features` function. This function iterates through trials and channels, calculates the Power Spectral Density (PSD) using `scipy.signal.welch`, and averages the power within predefined `bands` (Theta, Alpha, Beta, Gamma).
        *   **Spatial Features (CSP):** Used `mne.decoding.CSP` (`n_components=6`). CSP finds spatial filters (linear combinations of channels) that maximize the variance difference between the two classes ("Up" vs "Down"). Applying these filters transforms the data (`X_csp`), effectively creating "virtual channels" optimized for discrimination.
    *   **Caching:** Cached the extracted features (`X_freq`, `X_csp`) and the fitted `csp` object together, using a key derived from the input data's cache key and feature parameters. This avoids re-extracting features. Handled loading older cache formats if the CSP object wasn't saved previously.
    *   **Visualization:** Plotted the topography of the first few CSP filters using `mne.viz.plot_topomap` (#`figures/csp_patterns.png`). This helps visualize the spatial patterns the model found most discriminative. *Note: Required creating a proper MNE `info` object with channel names and a standard montage for plotting.*

4.  **Improvement 3: Feature Combination & Selection:**
    *   **ML-Rationale:** Combining different types of features (frequency power + CSP components) can provide a richer representation of the brain state. However, this increases dimensionality. Feature selection helps reduce dimensionality, prevent overfitting, and focus the model on the most relevant features.
    *   **Implementation:**
        *   Combined frequency and CSP features using `np.hstack`.
        *   Applied `SelectKBest` with the `f_classif` (ANOVA F-test) score function to select the top `k=50` features from the combined set. `k=50` was chosen as a reasonable number to reduce dimensionality while retaining significant information.
    *   **Caching:** Cached the selected features (`X_selected`) and their indices (`selected_indices`), using a key derived from the combined features' cache key.

5.  **Improvement 4: Advanced Classification Pipeline (Tuning & Ensemble):**
    *   **ML-Rationale:** Default classifier parameters are rarely optimal. Hyperparameter tuning finds better settings. Combining multiple diverse classifiers (ensemble) often yields more robust and accurate predictions than any single model.
    *   **Implementation:**
        *   **Train/Test Split:** Split the *selected* features (`X_selected`) into training (70%) and testing (30%) sets using `train_test_split` (`random_state=42` for reproducibility).
        *   **Hyperparameter Tuning:** Defined parameter grids (`svm_param_grid`, `rf_param_grid`, `lda_param_grid`) for SVM, Random Forest, and LDA. Used `GridSearchCV` with 5-fold cross-validation (`KFold`) on the *training set* to find the best hyperparameters for each classifier based on cross-validated accuracy.
        *   **Ensemble Model:** Created a `VotingClassifier` using the *best* estimators found by `GridSearchCV` for SVM, RF, and LDA. Used 'soft' voting (averaging probabilities). Trained the ensemble on the training set.
    *   **Caching:** Cached the trained `GridSearchCV` objects and the final `ensemble` model, using a key derived from the selected features' cache key and the training set shape.

6.  **Improvement 5: Evaluation & Comparison:**
    *   **ML-Rationale:** Need to evaluate the final models on unseen test data to get an unbiased estimate of performance. Comparing against a baseline quantifies the benefit of the improvements.
    *   **Implementation:**
        *   Predicted labels for the *test set* using the tuned individual models and the ensemble model.
        *   Calculated test accuracy for each model (`accuracy_score`).
        *   Generated a detailed `classification_report` (precision, recall, F1-score) and a confusion matrix (`seaborn.heatmap`, #`figures/confusion_matrix.png`) for the best performing model (ensemble).
        *   **Baseline Comparison:** Trained a simple baseline pipeline (StandardScaler + default SVM) on the *flattened, non-feature-engineered* data (`X_flat`). Calculated its test accuracy.
        *   Calculated and printed the accuracy improvement of the best advanced model over the baseline.
    *   **Visualization:** Created a bar chart comparing the test accuracies of the baseline, tuned SVM, RF, LDA, and the ensemble model against chance level (#`figures/model_comparison.png`).

**Outcome & Conclusion:**

*   The script successfully implements a multi-stage pipeline incorporating multi-subject analysis, advanced feature engineering (Frequency + CSP), feature selection, hyperparameter tuning, and ensemble modeling.
*   Caching was crucial for making iterative development feasible.
*   Visualizations helped understand the data and model behavior (average signals, CSP patterns, confusion matrix, accuracy comparison).
*   The final ensemble model's accuracy was compared against the baseline, demonstrating the effectiveness (or lack thereof, for the difficult Inner vs Inner case) of the implemented enhancements. For the specific "Inner Up" vs "Inner Down" comparison run in the script, the accuracy improvement over baseline was marginal (50.3% vs 50.0%), highlighting the difficulty of this specific task. However, the framework itself is sound and achieved much higher accuracies on easier comparisons when tested via the UI (`day1_ui.py`) or systematically in `day2.py`.

This script represents a significant step up from basic analysis, establishing a robust, reusable, and neuroscience-informed pipeline for EEG classification.

## Devlog Entry: `src/day2.py` - Systematic Pairwise Classification

**Date:** 2025-04-11

**Goal:** After establishing a robust classification pipeline in `day1.py` (and refining it in `src/day2_analysis_pipeline.py`), the goal of `day2.py` was to systematically evaluate the discriminability between *all possible pairs* of mental states present in the dataset. This provides a comprehensive map of which cognitive contrasts are easy or hard to decode using our current methodology.

**Thought Process & Rationale:**

*   **Beyond Single Comparisons:** `day1.py` focused on one specific comparison (e.g., Inner Up vs Inner Down). However, the dataset contains 3 conditions (Inner, Pronounced, Visualized) and 4 classes (Up, Down, Left, Right), resulting in 12 distinct mental states.
*   **Mapping Discriminability:** We wanted to know how well our pipeline performs on *all* 66 unique pairwise combinations of these 12 states (e.g., Inner-Up vs Pron-Down, Inner-Left vs Vis-Right, Pron-Up vs Pron-Down, etc.).
*   **Identifying Promising Pairs:** This systematic analysis helps identify which pairs yield high classification accuracy. These pairs are potentially better candidates for building a practical Brain-Computer Interface (BCI).
*   **Understanding Limitations:** Conversely, identifying pairs with low accuracy highlights the limitations of the current features or models for distinguishing those specific mental states.
*   **Leveraging the Enhanced Pipeline:** Instead of rewriting the analysis logic, `day2.py` imports and utilizes the refined, cached pipeline from `src/day2_analysis_pipeline.py`. This ensures consistency and leverages the improvements made (multi-subject, Freq+CSP features, SelectKBest, tuned ensemble model, caching).

**Implementation Steps:**

1.  **Setup & Configuration:**
    *   Imports necessary libraries (`itertools`, `time`, `numpy`, `os`, `datetime`, `pandas`).
    *   Imports the core analysis function `run_pairwise_analysis_enhanced` from `src/day2_analysis_pipeline.py`.
    *   Defines configuration: `subject_list` (1-10), `datatype` ("EEG"), `t_start`/`t_end` (1.5s-3.5s), output log (`day2_devlog.md`) and results (`day2_pairwise_results.csv`) filenames.

2.  **Generating State Pairs:**
    *   Defines all `conditions` ("Inner", "Pron", "Vis") and `classes` ("Up", "Down", "Left", "Right").
    *   Uses `itertools.product` to generate all 12 unique `(Condition, Class)` tuples (e.g., `('Inner', 'Up')`, `('Pron', 'Down')`, etc.).
    *   Uses `itertools.combinations` to generate all 66 unique pairs of these 12 states.

3.  **Systematic Analysis Loop:**
    *   Iterates through each of the 66 `state_pairs`.
    *   For each pair:
        *   Extracts the condition and class for state 1 and state 2.
        *   Formats these into the nested list structure required by `transform_for_classificator` (e.g., `Conditions = [['Inner'], ['Pron']]`, `Classes = [['Up'], ['Down']]`).
        *   Calls `run_pairwise_analysis_enhanced`, passing the current pair's conditions, classes, and other configuration parameters. This function handles:
            *   Loading/caching multi-subject data for the specific pair.
            *   Checking class balance.
            *   Extracting/caching Freq+CSP features.
            *   Selecting/caching top 50 features.
            *   Splitting data.
            *   Training/caching the tuned ensemble model (SVM, RF, LDA).
            *   Evaluating the ensemble on the test set.
            *   Returning a dictionary of metrics (accuracy, precision, recall, F1, trial counts, balance ratio).
        *   Records the time taken for the pair's analysis.
        *   Appends the results (or error message) immediately to the `day2_devlog.md` file for live progress tracking.
        *   Stores the detailed metrics dictionary in a `results_list`.
    *   Includes basic error handling (`try...except`) to log issues and continue with the next pair if one fails.

4.  **Logging and Saving Results:**
    *   Adds start and end timestamps to the `day2_devlog.md` file.
    *   After the loop finishes, converts the `results_list` (list of dictionaries) into a pandas DataFrame.
    *   Reorders columns for readability.
    *   Sorts the DataFrame by accuracy (descending).
    *   Saves the complete, sorted results table to `logs/day2_pairwise_results.csv`.

**Key Considerations/Limitations Mentioned in Code Comments:**

*   **Fixed Pipeline:** The *exact same* pipeline (features, selection k=50, model tuning strategy) is applied to all 66 pairs. This might not be the absolute optimal approach for every single contrast (e.g., maybe different features or `k` would be better for Vis-Up vs Vis-Down). However, it provides a consistent baseline for comparison across pairs.
*   **No Significance Testing:** The script focuses on calculating performance metrics. It does *not* perform statistical tests (like permutation testing) to determine if the accuracy for each pair is significantly above chance level. This would be a computationally intensive next step for specific pairs of interest.

**Outcome & Output:**

*   The script successfully ran the enhanced classification pipeline on all 66 pairwise combinations of mental states.
*   **`logs/day2_devlog.md`:** Contains a timestamped log of the run, showing the key metrics (Acc, P, R, F1, Balance, N) and time taken for each pair as it completed.
*   **`logs/day2_pairwise_results.csv`:** Provides a comprehensive summary table of all 66 pairs, including all calculated metrics, trial counts, balance ratio, and time taken, sorted by accuracy. This CSV file allows for easy analysis and identification of the most (and least) discriminable mental state pairs using the implemented pipeline.

This systematic analysis provides valuable insights into the dataset's structure and the capabilities of the classification approach across a wide range of cognitive contrasts.

## Summary of Pairwise Classification Results (`day2.py`)

The systematic analysis in `day2.py` evaluated the performance of the enhanced classification pipeline (Multi-subject, Freq+CSP features, SelectKBest, Tuned Ensemble) on all 66 unique pairs of the 12 mental states (3 Conditions x 4 Directions). The results reveal significant differences in discriminability depending on the nature of the comparison.

**Key Findings:**

1.  **High Discriminability Across Conditions:**
    *   The pipeline achieved the highest accuracies when distinguishing between different *conditions*, particularly **Inner Speech vs. Pronounced Speech**.
    *   *Examples (from `day2_pairwise_results.csv`):*
        *   `Inner-Left_vs_Pron-Up`: **90.1%** Accuracy
        *   `Inner-Up_vs_Pron-Down`: **89.7%** Accuracy
        *   `Inner-Left_vs_Pron-Down`: **89.3%** Accuracy
        *   Many other Inner vs. Pronounced pairs achieved accuracies in the **high 80s**.
    *   **Pronounced Speech vs. Visualized** states also showed high discriminability.
    *   *Examples (from `day2_pairwise_results.csv`):*
        *   `Pron-Left_vs_Vis-Up`: **86.7%** Accuracy
        *   `Pron-Up_vs_Vis-Down`: **87.1%** Accuracy
        *   Many Pronounced vs. Visualized pairs achieved accuracies in the **mid-to-high 80s**.
    *   **Interpretation:** The neural patterns associated with actually speaking (Pronounced) are significantly different from those associated with inner speech or visualization, making these distinctions relatively easy for the model.

2.  **Moderate Discriminability Between Inner Speech and Visualization:**
    *   Distinguishing between **Inner Speech and Visualized** states yielded moderate accuracies, generally better than chance but lower than comparisons involving Pronounced speech.
    *   *Examples (from `day2_pairwise_results.csv`):*
        *   `Inner-Right_vs_Vis-Left`: **77.6%** Accuracy
        *   `Inner-Up_vs_Vis-Left`: **77.3%** Accuracy
        *   `Inner-Down_vs_Vis-Left`: **75.8%** Accuracy
        *   Most Inner vs. Visualized pairs achieved accuracies in the **high 60s to mid 70s**.
    *   **Interpretation:** While distinct, the neural signatures of inner speech and visualization share more similarities (both being internal mental tasks) compared to overt speech, making them harder to separate perfectly.

3.  **Low Discriminability Within Conditions:**
    *   The most challenging task was distinguishing between different *directions* within the *same condition*.
    *   **Within Inner Speech:** Accuracies were consistently near chance level (50%).
        *   *Examples (from `day2_pairwise_results.csv`):*
            *   `Inner-Up_vs_Inner-Right`: **54.8%** Accuracy
            *   `Inner-Down_vs_Inner-Right`: **50.3%** Accuracy
            *   `Inner-Up_vs_Inner-Left`: **50.0%** Accuracy
            *   `Inner-Up_vs_Inner-Down`: **49.1%** Accuracy
            *   `Inner-Left_vs_Inner-Right`: **46.4%** Accuracy
            *   `Inner-Down_vs_Inner-Left`: **45.8%** Accuracy
    *   **Within Pronounced Speech:** Accuracies were slightly better than chance but still low.
        *   *Examples (from `day2_pairwise_results.csv`):*
            *   `Pron-Up_vs_Pron-Right`: **61.8%** Accuracy
            *   `Pron-Down_vs_Pron-Left`: **60.6%** Accuracy
            *   `Pron-Up_vs_Pron-Down`: **55.9%** Accuracy
    *   **Within Visualized:** Accuracies were also near chance level.
        *   *Examples (from `day2_pairwise_results.csv`):*
            *   `Vis-Up_vs_Vis-Down`: **53.8%** Accuracy
            *   `Vis-Down_vs_Vis-Left`: **50.9%** Accuracy
            *   `Vis-Up_vs_Vis-Left`: **48.2%** Accuracy
            *   `Vis-Up_vs_Vis-Right`: **46.5%** Accuracy
    *   **Interpretation:** This is the core challenge. The subtle differences in neural activity corresponding *only* to the *direction* of inner thought (or visualization, or even pronunciation) are very difficult to decode reliably using the current feature set (Frequency Bands + CSP) and models, even with multi-subject data. The dominant signal differences appear to be related to the *type* of mental task (Inner vs. Pronounced vs. Visualized) rather than the specific directional content within a task type.

**Overall Conclusion:**

The enhanced pipeline is highly effective at distinguishing between different *types* of mental tasks (Inner, Pronounced, Visualized). However, decoding the specific *content* (direction) within a single task type, especially inner speech, remains extremely challenging and achieves near-chance performance with the current methods. This highlights the subtlety of the neural correlates of directional inner speech and suggests that more advanced features, models, or perhaps different experimental paradigms might be needed to reliably decode this specific type of mental content.