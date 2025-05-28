# Audio Pattern Recognition

This repository contains the project developed for the MSc course **Audio Pattern Recognition** offered by the University of Milan.  
📘 Course Link: [Audio Pattern Recognition - UniMi](https://www.unimi.it/en/education/degree-programme-courses/2025/audio-pattern-recognition)

This project implements various machine learning approaches for music genre classification using audio features. The system extracts Mel-frequency cepstral coefficients (MFCCs) from audio files and employs different classification models to predict music genres.

## Project Overview

The project explores multiple approaches to music genre classification:
- MFCC Feature Extraction
- Random Forest Classification
- CNN-based Classification
- RNN-based Classification
- K-means Clustering for Genre Analysis

## Features

- Audio feature extraction using MFCCs
- Multiple classification approaches:
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Random Forest Classifier
  - K-means Clustering
- Comprehensive evaluation metrics and visualization
- Support for multiple music genres

## Requirements

- Python 3.x
- TensorFlow
- librosa
- scikit-learn
- numpy
- matplotlib
- seaborn

## Project Structure

- `mfcc_feature_extraction_genre_classifier.ipynb`: Extracts MFCC features from audio files
- `CNN_Classifier.ipynb`: Implements CNN-based genre classification
- `RNN_Classifier.ipynb`: Implements RNN-based genre classification
- `Random_Forest_Classifier.ipynb`: Implements Random Forest-based classification
- `Genre_Clustering_with_KMeans.ipynb`: Implements K-means clustering for genre analysis

## Dataset

The project uses the GTZAN dataset for music genre classification. You can download it from:
[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## Usage

1. Prepare your audio dataset in the following structure:
```
dataset/
    genre1/
        audio1.wav
        audio2.wav
        ...
    genre2/
        audio1.wav
        audio2.wav
        ...
    ...
```

2. Run the MFCC feature extraction notebook to process the audio files:
```python
python mfcc_feature_extraction_genre_classifier.ipynb
```

3. Choose and run one of the classification notebooks:
- For CNN-based classification: `CNN_Classifier.ipynb`
- For RNN-based classification: `RNN_Classifier.ipynb`
- For Random Forest classification: `Random_Forest_Classifier.ipynb`
- For genre clustering: `Genre_Clustering_with_KMeans.ipynb`

## Model Details

### CNN Classifier
- Input: MFCC features (130x13x1)
- Architecture:
  - 3 Convolutional layers with batch normalization
  - MaxPooling layers
  - Dense layers with dropout
  - Softmax output layer

### RNN Classifier
- Processes sequential audio features
- Suitable for capturing temporal patterns in music

### Random Forest Classifier
- Ensemble learning approach
- Handles non-linear relationships in audio features

### K-means Clustering
- Unsupervised learning approach
- Groups similar music patterns together

## Techniques Used

- **Python** (with libraries like `librosa`, `numpy`, `scikit-learn`, `matplotlib`)
- **Signal Processing** fundamentals
- **Supervised Learning** for audio classification
- **Spectrogram Analysis**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Moein Ghaeini
