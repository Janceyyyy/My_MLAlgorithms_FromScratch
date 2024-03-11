# Machine Learning Algorithms

This repository is a comprehensive collection of machine learning algorithms developed from the ground up using Python and NumPy. The implementations are inspired by the author's work on assignments from the CSCI1420 Machine Learning course and the DATA2060 Machine Learning: from Theory to Algorithms course at Brown University. This collection covers a wide range of models, from simple linear regressions to more complex structures like Support Vector Machines and Kmeans Clustering, applied across various datasets for both classification and regression challenges.

## Running the Models

To execute any of the machine learning models in this collection:

1. Navigate to the model's specific directory.
2. Execute `main.py` within that directory. This script utilizes functions defined in `models.py` and possibly other supportive scripts to apply the models to specific datasets, displaying the model's performance in the standard output.

The repository includes a variety of models, each with its own directory containing all necessary scripts for execution. Additionally, some models generate plots and diagrams, which are described in the output section of this README.

## Environment and Dependencies

The implementations rely on several key Python packages:

- Python version: 3.10.7
- NumPy version: 1.23.3
- Matplotlib version: 3.6.0
- Pandas version: 1.4.2
- Quadprog version: 0.1.11
- Scikit-learn version: 1.1.1

## Overview of Implemented Models

Below is a summary of each model, including datasets used, key functionalities, and how to interpret the results directly from the output descriptions in this README.

### Linear Regression

- **Dataset:** Wine quality dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
- **Functionality:** Implements training, prediction, and MSE calculation for linear regression.
- **Usage:** Predict wine quality ratings based on 11 attributes. 
- **Results:** Outputs include training and testing MSE directly printed to stdout.

### Logistic Regression

- **Dataset:** UCI Census Income data (1994) available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income).
- **Functionality:** Implements logistic regression training, prediction, and accuracy calculation.
- **Usage:** Predicts education levels based on census attributes.
- **Results:** Outputs epoch loss, number of epochs, and test accuracy.

### Validation and Regularization

- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) Data [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
- **Functionality:** Evaluates models using train/test/validation splits and k-fold cross-validation for various lambda values.
- **Usage:** Predicts breast cancer presence.
- **Results:** Train and validation accuracy, along with k-fold validation errors, are directly reported.

### Decision Trees

- **Datasets:** Spambase and Chess datasets from [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
- **Functionality:** Implements binary classification decision trees with splitting and pruning.
- **Usage:** Classifies emails as spam and predicts chess game outcomes.
- **Results:** Training and test accuracy of pruned and unpruned trees for various parameters are printed.

### Naive Bayes

- **Dataset:** German Credit dataset [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
- **Functionality:** Implements Naive Bayes with fairness assessment methods.
- **Usage:** Predicts creditworthiness based on demographic and financial attributes.
- **Results:** Outputs include train and test accuracy, alongside fairness measures.

### Support Vector Machine (SVM)

- **Dataset:** Spambase dataset [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase).
- **Functionality:** Implements SVM for binary classification using quadprog for optimization.
- **Usage:** Classifies emails as spam.
- **Results:** Outputs training and testing accuracy for linear, RBF, and polynomial kernels.

### Kmeans Clustering

- **Dataset:** Hand-written digits, each row in `digits.csv` representing a digit with 64 pixel values.
- **Functionality:** Trains and predicts using Kmeans clustering.
- **Usage:** Clusters hand-written digit images.
- **Results:** Model accuracy and cluster centers resembling digits are directly reported.


## Acknowledgments

The code and approaches in this repository are adapted from coursework at Brown University, specifically CSCI1420 Machine Learning and DATA2060 Machine Learning: from Theory to Algorithms. These implementations are intended for educational purposes, aiming to provide hands-on experience with machine learning concepts. All original material and rights are attributed to Brown University.