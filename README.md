# News Article Classification Project

## Project Overview
This project implements a complete machine learning pipeline to classify news articles into categories using Python scripts only. The solution follows a structured folder architecture and runs entirely from the terminal.

## Dataset Source
Dataset: 20 Newsgroups Dataset  
Source: Scikit-learn built-in dataset  
Link: https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset  

For better classification performance, four distinct categories were selected:
- talk.politics.misc
- rec.sport.baseball
- sci.med
- comp.graphics

## Project Architecture

news_classification_project/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ └── config.py
│
├── models/
│ └── news_classifier.pkl
│
├── results/
│ └── metrics.txt
│
├── requirements.txt
├── README.md
└── main.py


## Machine Learning Pipeline

1. Data Preprocessing  
   - Loaded dataset using sklearn
   - Removed headers, footers, and quotes

2. Feature Engineering  
   - TF-IDF Vectorization
   - Unigrams and bigrams
   - 10,000 max features

3. Model Training  
   - Logistic Regression
   - max_iter=2000

4. Model Evaluation  
   - Accuracy Score
   - Confusion Matrix
   - Results saved in results/metrics.txt

## Model Used
Logistic Regression with TF-IDF features.

## Final Results
Final Accuracy: ~88%

## How to Run the Project

1. Install dependencies:
pip install -r requirements.txt


2. Run the full pipeline:
python main.py


The model and evaluation metrics will be saved automatically.

## Key Learnings
- Built a structured ML pipeline without using Jupyter Notebook
- Implemented text preprocessing and TF-IDF feature extraction
- Trained and evaluated a multi-class classification model
- Saved model artifacts and metrics for reproducibility

## GitHub Repository
(Add your GitHub link here)

## Video Explanation
(Add your video link here)