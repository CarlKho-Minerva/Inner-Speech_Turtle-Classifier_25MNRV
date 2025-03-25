# Understanding Classification Methods for EEG-Based Thought Recognition

## Our Current Approach: SVM with RBF Kernel

When processing our Inner Speech Dataset to distinguish between "thinking Up" and "thinking Down", we use a two-step pipeline:

### 1. StandardScaler: Normalizing Brain Activity

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**What this does to our EEG data:**

- Calculates the mean and standard deviation of each feature (channel-timepoint combination)
- Subtracts the mean from each value (centering the data)
- Divides by the standard deviation (scaling to unit variance)

**Real example:** If electrode Cz shows higher baseline activity (+15μV) than electrode Oz (+2μV), standardization ensures both contribute equally to the classification.

### 2. Support Vector Machine (RBF Kernel)

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)
```

**What this does to our EEG data:**

- Calculates similarity (kernel) values between all pairs of trials
- Maps our 65,536-dimensional data into an infinite-dimensional space
- Finds an optimal hyperplane that separates "Up" from "Down" trials
- Identifies "support vectors" (critical trials that define the boundary)

**Real example:** SVM might discover that the pattern of activity in motor cortex at 2.3 seconds after cue presentation is crucial for distinguishing "Up" from "Down" thoughts.

## Alternative Classification Approaches in Detail

### 1. Linear Discriminant Analysis (LDA)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
```

**What LDA does to our EEG data:**

- Computes the mean EEG pattern for "Up" trials and "Down" trials
- Calculates the shared covariance matrix (how features vary together)
- Creates a projection that maximizes between-class separation while minimizing within-class scatter
- New trials are classified based on which class mean they're closer to (after projection)

**Real example:** LDA might create a single discriminant dimension where positive values indicate "Up" thoughts and negative values indicate "Down" thoughts. The weights might emphasize electrodes over motor cortex, particularly in the alpha (8-12Hz) range.

### 2. Random Forests

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

**What Random Forests do to our EEG data:**

- Creates 100 different decision trees
- Each tree examines a random subset of ~256 time-channel features
- Each node in a tree splits data based on questions like "Is channel FC3 at 2.1s > 3.4μV?"
- Combines all trees' votes for the final prediction
- Automatically ranks features by importance

**Real example:** Random Forest might discover that electrodes C3 and C4 (over motor cortex) at 2.0-2.5s after cue are most important for classification, with tree splits occurring at different voltage thresholds.

### 3. Convolutional Neural Networks (CNNs)

```python
import tensorflow as tf

# Reshape data to preserve spatial structure [trials, channels, timepoints, 1]
X_reshaped = X.reshape(X.shape[0], 128, 512, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (1, 10), activation='relu'), # Temporal convolution
    tf.keras.layers.Conv2D(32, (128, 1), activation='relu'), # Spatial convolution
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_reshaped, y)
```

**What CNNs do to our EEG data:**

- Preserves spatial (electrode) and temporal (time sample) relationships
- Applies filters that slide across time to detect temporal patterns
- Applies filters that integrate across electrodes to detect spatial patterns
- Automatically learns which patterns distinguish "Up" from "Down" thoughts

**Real example:** A CNN might learn filters that detect the specific rhythm in motor cortex that changes when thinking "Up" versus "Down", automatically discovering that beta rhythm (13-30Hz) decreases more during "Down" imagery than "Up" imagery.

### 4. Common Spatial Patterns (CSP)

```python
from mne.decoding import CSP

# Reshape back to [trials, channels, timepoints]
X_reshaped = X.reshape(X.shape[0], 128, 512)

# Apply CSP
csp = CSP(n_components=6, reg=None, log=True)
X_csp = csp.fit_transform(X_reshaped, y)

# Use simple classifier on CSP features
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_csp, y)
```

**What CSP does to our EEG data:**

- Designs spatial filters that maximize variance for one class while minimizing it for the other
- Transforms 128-channel data into just 6 virtual channels that optimally separate classes
- These virtual channels represent specific brain patterns that differ between "Up" and "Down" thoughts
- Dramatically reduces features from 65,536 to just 6

**Real example:** CSP might create spatial filters focusing on central electrodes (C3, Cz, C4), extracting patterns that show increased synchronization in the left motor cortex for "Up" thoughts versus right motor cortex for "Down" thoughts.

### 5. Riemannian Geometry Classifier

```python
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

# Convert trials to covariance matrices
cov = Covariances().transform(X_reshaped)

# Minimum Distance to Mean classifier in Riemannian space
mdm = MDM(metric='riemann')
mdm.fit(cov, y)
```

**What Riemannian approaches do to our EEG data:**

- Transforms each trial into a 128×128 covariance matrix (capturing relationships between all electrode pairs)
- Treats these matrices as points on a curved manifold rather than flat Euclidean space
- Uses specialized distance metrics that account for the curvature of the space
- Classifies based on which class's geometric mean is closer

**Real example:** The covariance matrix might reveal that "Up" thoughts create stronger correlations between frontal and central electrodes, while "Down" thoughts increase correlations between central and parietal regions—a pattern the Riemannian classifier can detect without explicit feature extraction.

## Advanced Feature Extraction Example: Frequency Domain

Instead of using raw signals, we could extract frequency information:

```python
from scipy import signal

def extract_frequency_features(X, fs=256):
    # Reshape to [trials, channels, timepoints]
    X_reshaped = X.reshape(X.shape[0], 128, 512)

    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    # Create empty feature array
    X_freq = np.zeros((X_reshaped.shape[0], X_reshaped.shape[1] * len(bands)))

    # Extract band power for each trial and channel
    for i in range(X_reshaped.shape[0]):
        for j in range(X_reshaped.shape[1]):
            signal_psd, freqs = signal.welch(X_reshaped[i, j], fs=fs)

            # Extract power in each band
            feature_idx = j * len(bands)
            for k, (band, (low, high)) in enumerate(bands.items()):
                band_power = np.mean(signal_psd[(freqs >= low) & (freqs <= high)])
                X_freq[i, feature_idx + k] = band_power

    return X_freq
```

**What this does to our EEG data:**

- Calculates power spectral density (PSD) for each channel
- Extracts average power in 5 frequency bands for each channel
- Transforms data from [500 trials × 65,536 features] to [500 trials × 640 features]
- Focuses specifically on brain rhythms rather than raw voltage changes

**Real example:** This would reveal that "Up" thoughts might produce stronger beta band (12-30Hz) suppression in central electrodes (indicating motor imagery), while "Down" thoughts might show more alpha (8-12Hz) power in parietal regions.

## Why Our Classification Choice Matters

The classification approach determines not just accuracy, but what we learn about brain function:

- **SVMs** tell us if patterns exist but not necessarily what they are
- **Random Forests** can reveal which specific time points and channels matter most
- **CSP** extracts the actual spatial patterns that differ between thought conditions
- **CNN** can discover complex spatiotemporal features we might not know to look for
- **Frequency analysis** reveals which brain rhythms are involved in different thoughts

For a brain-computer interface, the right choice depends on your priorities: accuracy, interpretability, computational efficiency, or adaptability to new users.
