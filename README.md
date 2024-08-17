
# K-Nearest Neighbors Classifier with Hyperparameter Tuning

This project demonstrates the use of the K-Nearest Neighbors (KNN) classifier for binary classification on a bank dataset. The project involves data preprocessing, model training, and hyperparameter tuning using Grid Search and Randomized Search.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Grid Search](#grid-search)
  - [Randomized Search](#randomized-search)
- [Results](#results)
- [Interpretations](#interpretations)

## Overview

This project utilizes the K-Nearest Neighbors (KNN) algorithm to classify bank customers based on various features. The target variable is binary, indicating whether a customer has subscribed to a term deposit. The project involves splitting the dataset into training and testing sets, training the KNN model, and improving model performance through hyperparameter tuning.

## Dataset

The dataset used in this project is a CSV file containing the following columns:

- `age`: Age of the customer
- `balance`: Balance of the customer's bank account
- `day`: Day of the month when the customer was last contacted
- `duration`: Duration of the last contact
- `campaign`: Number of contacts performed during this campaign
- `pdays`: Number of days since the customer was last contacted in a previous campaign
- `previous`: Number of contacts performed before this campaign
- `y`: Target variable (0 = No, 1 = Yes)

## Requirements

The following Python libraries are required to run the project:

- pandas
- numpy
- matplotlib
- scikit-learn

Install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

To run the project, simply execute the script in a Python environment.

```bash
python knn_classifier.py
```

## Model Training and Evaluation

The KNN model is trained on the training dataset using the default parameters. The model's performance is evaluated on the test dataset using classification metrics such as precision, recall, f1-score, and accuracy.

## Hyperparameter Tuning

To improve the model's performance, hyperparameter tuning is performed using two methods: Grid Search and Randomized Search.

### Grid Search

Grid Search is used to explore a specified range of hyperparameters. The following hyperparameters are tuned:

- `n_neighbors`: Number of neighbors to use
- `weights`: Weight function used in prediction
- `leaf_size`: Leaf size passed to BallTree or KDTree

```python
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'weights': ['uniform', 'distance'],
    'leaf_size': [10, 50, 100, 500]
}
```

### Randomized Search

Randomized Search is used to explore a larger range of hyperparameters with a specified number of iterations. The following hyperparameters are tuned:

- `n_neighbors`: Number of neighbors to use
- `weights`: Weight function used in prediction
- `leaf_size`: Leaf size passed to BallTree or KDTree

```python
param_grid = {
    'n_neighbors': np.arange(1, 20, 2),
    'weights': ['uniform', 'distance'],
    'leaf_size': np.arange(1, 500)
}
```

## Results

The best hyperparameter combinations found using Grid Search and Randomized Search are:

- **Grid Search**: `{'leaf_size': 10, 'n_neighbors': 17, 'weights': 'distance'}`
- **Randomized Search**: `{'weights': 'distance', 'n_neighbors': 19, 'leaf_size': 243}`

The overall accuracy of the model improved from `0.87` to `0.89` after hyperparameter tuning.

## Interpretations

The best hyperparameter combination improved the overall accuracy of the model. However, if recall of the positive class is the metric of interest, the adjusted hyperparameters performed worse compared to the untuned model. This highlights the trade-off between different performance metrics during model tuning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

