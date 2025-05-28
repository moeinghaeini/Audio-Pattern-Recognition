# Audio Pattern Recognition

This repository contains the project developed for the MSc course **Audio Pattern Recognition** offered by the University of Milan.  
📘 Course Link: [Audio Pattern Recognition - UniMi](https://www.unimi.it/en/education/degree-programme-courses/2025/audio-pattern-recognition)

## 📋 Project Overview

This project implements various machine learning approaches for music genre classification using audio features. The system extracts Mel-frequency cepstral coefficients (MFCCs) from audio files and employs different classification models to predict music genres.

### 🎯 Key Features

- **Audio Processing**
  - MFCC feature extraction from audio files
  - Audio signal preprocessing and normalization
  - Spectrogram generation and analysis

- **Multiple Classification Approaches**
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Random Forest Classifier
  - K-means Clustering

- **Analysis & Visualization**
  - Comprehensive evaluation metrics
  - Performance comparison between models
  - Feature importance analysis
  - Clustering visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Audio-Pattern-Recognition.git
cd Audio-Pattern-Recognition
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages

- TensorFlow 2.x
- librosa
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- jupyter

## 📁 Project Structure

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

## 📊 Dataset

The project uses the GTZAN dataset for music genre classification. The dataset contains 1000 audio tracks, each 30 seconds long, across 10 different genres.

### Dataset Structure
```
dataset/
    blues/
        audio1.wav
        audio2.wav
        ...
    classical/
        audio1.wav
        audio2.wav
        ...
    ...
```

Download the dataset from: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## 🎮 Usage

### 1. Feature Extraction

Run the MFCC feature extraction notebook to process the audio files:
```bash
jupyter notebook mfcc_feature_extraction_genre_classifier.ipynb
```

This will:
- Process all audio files in the dataset
- Extract MFCC features
- Save processed data to `Data.json`
- Generate K-means specific data in `Data_kmeans.json`

### 2. Model Training

Choose and run one of the classification notebooks:

- **CNN Classification**:
```bash
jupyter notebook CNN_Classifier.ipynb
```

- **RNN Classification**:
```bash
jupyter notebook RNN_Classifier.ipynb
```

- **Random Forest Classification**:
```bash
jupyter notebook Random_Forest_Classifier.ipynb
```

- **Genre Clustering**:
```bash
jupyter notebook Genre_Clustering_with_KMeans.ipynb
```

## 🧠 Model Details

### CNN Classifier
- **Input**: MFCC features (130x13x1)
- **Architecture**:
  - 3 Convolutional layers with batch normalization
  - MaxPooling layers
  - Dense layers with dropout
  - Softmax output layer
- **Use Case**: Best for capturing spatial patterns in audio features

### RNN Classifier
- **Input**: Sequential MFCC features
- **Architecture**:
  - LSTM/GRU layers
  - Dense layers
  - Softmax output
- **Use Case**: Ideal for capturing temporal patterns in music

### Random Forest Classifier
- **Features**: Statistical features from MFCCs
- **Advantages**:
  - Handles non-linear relationships
  - Feature importance analysis
  - Less prone to overfitting
- **Use Case**: Good baseline model for comparison

### K-means Clustering
- **Input**: Processed features from `Data_kmeans.json`
- **Purpose**: Unsupervised genre discovery
- **Use Case**: Finding natural groupings in music patterns

## 📈 Performance Metrics

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC curves (where applicable)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Moein Ghaeini

## 🙏 Acknowledgments

- University of Milan for the course materials
- GTZAN dataset creators
- Open-source community for the amazing libraries used in this project
