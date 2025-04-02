# GenreSense: Film Classification through Natural Language Understanding

## Project Overview
GenreSense is a movie genre classification system that uses natural language processing techniques to analyze movie descriptions and predict their genres. This project implements multiple machine learning algorithms from scratch, including K-Nearest Neighbors (KNN) and Naive Bayes, to classify movies into 23 different genres based on their textual descriptions.

## Key Features
- **Text Processing Pipeline**: Custom implementation of text preprocessing functions including tokenization, stemming, lemmatization, and linguistic modulation
- **Vectorization Techniques**: Implementation of Bag-of-Words and TF-IDF vectorization from scratch
- **Classification Algorithms**: Custom implementations of:
  - K-Nearest Neighbors with different distance metrics (cosine similarity and Euclidean distance)
  - Naive Bayes classifier for text classification
- **Evaluation Metrics**: Computation of accuracy, precision, recall, and F1-score to evaluate model performance

## Dataset
The project works with a movie dataset containing titles, descriptions, and genre labels. The dataset is split into training and testing sets to evaluate the model's performance.

## Results
- The KNN classifier achieved 84% accuracy across 23 different movie genres
- Performance varies across genres, with comedy (97% F1-score) and drama (92% F1-score) showing the best results
- Visualization of genre distribution and model performance is included

## Technical Implementation
- Built entirely in Python using fundamental libraries (NumPy, Pandas, Matplotlib)
- All machine learning algorithms implemented from scratch (no sklearn implementations)
- Custom implementation of TF-IDF and Bag-of-Words vectorization

## Future Improvements
- Experiment with ensemble methods to improve classification accuracy
- Implement additional distance metrics for KNN
- Explore deep learning approaches like word embeddings and neural networks

## Author
Amirsepehr Abedini - 400243056
