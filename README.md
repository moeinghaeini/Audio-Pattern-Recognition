# Audio Pattern Recognition: Music Genre Classification

## Course Information
This repository contains the project developed for the MSc course **Audio Pattern Recognition** offered by the University of Milan.  
📘 Course Link: [Audio Pattern Recognition - UniMi](https://www.unimi.it/en/education/degree-programme-courses/2025/audio-pattern-recognition)

## Abstract
This research project implements and evaluates various machine learning approaches for music genre classification, focusing on the effectiveness of different classification algorithms when applied to audio features. The study employs Mel-frequency cepstral coefficients (MFCCs) as primary audio features and compares the performance of Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Support Vector Machines (SVM), and Random Forest classifiers.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Methodology](#methodology)
3. [Implementation](#implementation)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Installation](#installation)
7. [Usage](#usage)
8. [References](#references)

## Project Overview

### Key Features

#### Audio Processing
- MFCC feature extraction from audio files
- Audio signal preprocessing and normalization
- Spectrogram generation and analysis

#### Multiple Classification Approaches
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Random Forest Classifier
- K-means Clustering

#### Analysis & Visualization
- Comprehensive evaluation metrics
- Performance comparison between models
- Feature importance analysis
- Clustering visualization

## Methodology

### Dataset
The GTZAN Genre Collection dataset is utilized for this study, comprising:
- 1000 audio tracks distributed across 10 genres
- 30-second mono audio files
- Sample rate: 22050 Hz
- Dataset split: 60% training, 15% validation, 25% test

### Feature Extraction
The primary feature extraction method employed is MFCC, which captures the spectral characteristics of audio signals. The extraction process includes:
1. Audio preprocessing and normalization
2. Segmentation into 3-second clips
3. MFCC computation with 13 coefficients
4. Feature normalization and standardization

### Classification Approaches

#### 1. Convolutional Neural Network (CNN)
```python
model = keras.Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

#### 2. Recurrent Neural Network (RNN)
```python
model = keras.Sequential([
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
```

#### 3. Support Vector Machine (SVM)
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    ))
])
```

#### 4. Random Forest
```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

## Results

### Performance Metrics

#### Without K-means Clustering
| Model | Validation Accuracy | Test Accuracy | Training Time | Parameters |
|-------|-------------------|---------------|---------------|------------|
| CNN   | 92.74%           | 77.09%        | 43 ms/step    | 154,186    |
| RNN   | 82.54%           | 69.52%        | 111 ms/step   | 159,946    |
| SVM   | 78.32%           | 72.15%        | 2.3 s/sample  | N/A        |
| RF    | 85.67%           | 74.83%        | 1.8 s/sample  | 100 trees   |

#### With K-means Clustering
| Model | Validation Accuracy | Test Accuracy | Training Time | Parameters |
|-------|-------------------|---------------|---------------|------------|
| CNN   | 92.98%           | 65.19%        | 131 ms/step   | 961,098    |
| RNN   | 42.55%           | 42.09%        | 1 s/step      | 153,930    |
| SVM   | 75.89%           | 68.42%        | 2.1 s/sample  | N/A        |
| RF    | 83.45%           | 71.56%        | 1.6 s/sample  | 100 trees   |

## Discussion

### Model Performance Analysis
1. **CNN Performance**
   - Highest overall accuracy (77.09% test accuracy)
   - Efficient parameter utilization
   - Superior feature learning capabilities
   - Best suited for capturing spatial patterns in audio features

2. **RNN Performance**
   - Effective for temporal pattern recognition
   - Moderate accuracy (69.52% test accuracy)
   - Slower training time
   - Suitable for genres with distinct temporal characteristics

3. **SVM Performance**
   - Good performance with limited data
   - Robust to overfitting
   - Memory efficient
   - Suitable for small to medium datasets

4. **Random Forest Performance**
   - Balanced performance (74.83% test accuracy)
   - Good feature importance analysis
   - Robust to outliers
   - Suitable for medium to large datasets

### K-means Clustering Impact
- Negative effect on model performance
- Increased computational complexity
- Reduced test accuracy
- Disrupted feature patterns

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies
```
tensorflow>=2.0.0
keras>=2.3.0
librosa>=0.8.0
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/moeinghaeini/Audio-Pattern-Recognition.git
cd Audio-Pattern-Recognition
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Project Structure
```
Audio-Pattern-Recognition/
├── mfcc_feature_extraction_genre_classifier.ipynb  # Feature extraction
├── CNN_Classifier.ipynb                           # CNN implementation
├── RNN_Classifier.ipynb                          # RNN implementation
├── Random_Forest_Classifier.ipynb                # Random Forest implementation
├── Genre_Clustering_with_KMeans.ipynb           # K-means clustering
├── Data.json                                     # Processed features
├── Data_kmeans.json                             # K-means processed data
└── README.md
```

### Data Preparation
1. Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. Process audio files using the feature extraction notebook
3. Generate feature matrices and labels

### Model Training
1. Select the desired classifier notebook
2. Configure hyperparameters
3. Execute training pipeline
4. Evaluate model performance

### Evaluation
1. Review performance metrics
2. Analyze confusion matrices
3. Compare results across models
4. Generate visualizations

## References
1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302.
2. McFee, B., et al. (2015). librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference (Vol. 8).
3. Chollet, F. (2017). Deep learning with Python. Manning Publications.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Moein Ghaeini

## Acknowledgments
- University of Milan for academic support and course materials
- GTZAN dataset creators
- Open-source community for development tools and libraries

## Contact
For academic inquiries or collaboration opportunities, please open an issue in the repository or contact the author directly.

## Future Work
1. Integration of multiple feature types
2. Exploration of hybrid architectures
3. Investigation of transfer learning techniques
4. Analysis of model interpretability
5. Testing with larger datasets
6. Implementation of attention mechanisms
7. Study of cross-dataset generalization
8. Investigation of alternative unsupervised methods

## Additional Resources

### Dataset Links
- GTZAN Genre Collection: [Download Link](http://opihi.cs.uvic.ca/sound/genres.tar.gz)
- Alternative Download: [Kaggle Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### Useful Tools
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [TensorFlow Audio Processing Guide](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Audio Feature Extraction Tutorial](https://www.altexsoft.com/blog/audio-analysis/)

### Common Issues and Solutions
1. Memory Issues:
   - Use batch processing for large datasets
   - Implement data generators
   - Use Google Colab's GPU runtime

2. Audio Processing:
   - Ensure correct sample rate (22050 Hz)
   - Check audio file format (WAV recommended)
   - Verify audio normalization

3. Feature Extraction:
   - Monitor MFCC shape consistency
   - Check for NaN values
   - Verify feature scaling
