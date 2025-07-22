Student Performance Prediction using Machine Learning
This project focuses on predicting the academic performance of students using real-world data through a full machine learning pipeline. It was developed as part of a final-year task to demonstrate the implementation of core ML techniques including preprocessing, feature engineering, EDA, model training, testing, and evaluation.

📌 Project Objective
To understand and implement the end-to-end workflow of a machine learning project by:

Preparing and analyzing real-world data

Building classification models

Evaluating model performance

Testing the model on real and synthetic data

📁 Dataset Used
Name: Student Performance Dataset

Source: UCI Machine Learning Repository

Records: 649

Features: Demographics, academic background, family conditions, lifestyle

Target Variable: Performance category (Low, Medium, High) derived from final exam score (G3)

⚙️ Workflow Summary
Data Preprocessing

Removed irrelevant columns

Encoded categorical features (e.g., sex, internet, activities)

Created performance label based on final grade

Binned age into ranges

Exploratory Data Analysis (EDA)

Used statistical summaries and visualizations

Generated bar plots, pair plots, and heatmaps

Identified strong correlations with G1, G2, studytime

Feature Selection

Retained all meaningful features based on correlation analysis

Model Building

Trained two models:

Logistic Regression

Support Vector Machine (SVM)

Evaluation

Used metrics: Accuracy, Precision, Recall, F1-Score

Confusion matrix used for error analysis

Logistic Regression achieved perfect accuracy on structured data

Testing

Evaluated on both structured test samples and random synthetic data

Structured data results: very high accuracy

Synthetic data results: lower accuracy (suggests need for better generalization)

🧠 Key Learnings
Learned full ML workflow from real-world data

Understood how feature correlations can improve model performance

Observed model performance drop on synthetic/unseen data

Reinforced importance of careful preprocessing and EDA
