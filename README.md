# Breast Cancer Classification using Gaussian Naive Bayes & PCA

A machine learning project that demonstrates how **Principal Component Analysis (PCA)** can be combined with a **Gaussian Naive Bayes classifier** to classify breast cancer cases.

The project focuses on understanding the complete machine learning workflow — from exploratory data analysis and preprocessing to dimensionality reduction, model training, and performance evaluation.

## Project Overview

The dataset contains diagnostic measurements related to breast cancer.

The objective is to build a classification model that predicts whether a tumor is:

* **M — Malignant**
* **B — Benign**

The notebook implements a complete ML pipeline using Python and Scikit-learn.

## Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train / Test Split
   ↓
Feature Standardization
   ↓
PCA Dimensionality Reduction
   ↓
Gaussian Naive Bayes
   ↓
Predictions
   ↓
Model Evaluation
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Machine Learning Techniques

### 1. Exploratory Data Analysis

The dataset is explored using:

* Dataset information
* Class distribution
* Descriptive statistics
* Histograms
* Skewness analysis
* Feature correlation matrices
* Correlation heatmaps

This helps understand the structure and relationships within the data.

### 2. Data Cleaning

The notebook removes unnecessary columns such as:

* `id`
* `Unnamed: 32`

The `diagnosis` column is used as the target variable.

### 3. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A fixed `random_state=42` is used for reproducibility, and `stratify=y` maintains the class distribution between training and testing sets.

### 4. Feature Scaling

`StandardScaler` is applied to standardize the numerical features before dimensionality reduction.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Principal Component Analysis

PCA is used to reduce the dimensionality of the dataset while retaining **95% of the variance**.

```python
pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test)
```

This helps reduce the number of features while preserving most of the information contained in the original dataset.

### 6. Gaussian Naive Bayes

After PCA transformation, a Gaussian Naive Bayes classifier is trained.

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train_pca, y_train)
```

Predictions and probabilities are then generated:

```python
y_pred = model.predict(X_test_pca)
y_prob = model.predict_proba(X_test_pca)
```

## Model Evaluation

The model is evaluated using:

### Accuracy

Measures the overall proportion of correctly classified observations.

```python
accuracy_score(y_test, y_pred)
```

### Confusion Matrix

The confusion matrix provides a breakdown of:

* True Positives
* True Negatives
* False Positives
* False Negatives

```python
confusion_matrix(y_test, y_pred)
```

### ROC-AUC

ROC-AUC is calculated using the predicted probability of the positive class.

```python
roc_auc_score(y_test, y_prob[:, 1])
```

## Key Learning Outcomes

Through this project, I explored:

* Exploratory Data Analysis
* Data cleaning
* Feature selection
* Train-test splitting
* Feature standardization
* Dimensionality reduction using PCA
* Gaussian Naive Bayes
* Classification
* Confusion matrix
* ROC-AUC evaluation
* Building an end-to-end machine learning workflow

## Project Structure

```text
Breast-Cancer-Classification/
│
├── Bayes_Theorem.ipynb
├── Breast_cancer.csv
└── README.md
```

> Make sure the dataset path in the notebook is updated when running the project on another machine. The current notebook uses a local Windows path for the CSV file.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Breast-Cancer-Classification
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Bayes_Theorem.ipynb
```

and run the cells sequentially.

## Future Improvements

Possible improvements include:

* Comparing Gaussian Naive Bayes with Logistic Regression, SVM, Random Forest, and KNN
* Hyperparameter tuning
* Cross-validation
* Precision, recall, and F1-score analysis
* ROC curve visualization
* Feature importance analysis
* Building a simple prediction web application

## Conclusion

This project demonstrates how dimensionality reduction and probabilistic classification can work together in a practical machine learning problem.

Using **PCA before Gaussian Naive Bayes** provides a useful example of how preprocessing and model selection can be combined to create a complete classification pipeline.

---

### Author

**Abhinav Anil Darade**

If you found this project useful, feel free to ⭐ the repository and connect with me on LinkedIn.

