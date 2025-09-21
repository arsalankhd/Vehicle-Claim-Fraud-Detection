# 🚗 Vehicle Claim Fraud Detection

## 📝 Overview

This project focuses on building and evaluating machine learning models to detect fraudulent vehicle insurance claims. Using a real-world dataset, the primary goal is to create a robust classification model that can accurately distinguish between fraudulent and legitimate claims. A key challenge in this dataset is the class imbalance, which is addressed using the SMOTE (Synthetic Minority Over-sampling Technique).

The entire process, from data exploration to model evaluation, is detailed in the `fraud_detection_model.ipynb` notebook.

---

## 📊 Dataset

The project uses the **Vehicle Claim Fraud Detection** dataset, available on [Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection). The dataset contains various features related to the policyholder, the vehicle, and the claim itself.

- **Target Variable**: `FraudFound_P` (1 for fraudulent, 0 for legitimate).
- The dataset is highly imbalanced, with fraudulent claims representing a small minority of the cases.

---

## 🛠️ Project Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Conducted an initial analysis of the features to understand their distributions and relationships.
- Visualized the class imbalance of the target variable.

### 2. **Data Preprocessing & Feature Engineering**
- **Categorical Encoding**: Applied one-hot encoding to convert categorical features (like `Month`, `DayOfWeek`, `Make`, etc.) into a numerical format suitable for machine learning models.
- **Data Splitting**: Divided the dataset into training (80%) and testing (20%) sets.

### 3. **Handling Class Imbalance with SMOTE**
- To address the severe class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied **only to the training data**.
- This technique generates synthetic samples for the minority class (fraudulent claims), creating a balanced training set and helping the models learn the patterns of fraud more effectively.

### 4. **Model Training and Evaluation**
Several classification models were trained on the balanced dataset and evaluated on the original, imbalanced test set:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVC)**
- **Decision Tree**
- **Random Forest**

The models were evaluated using metrics appropriate for imbalanced classification:
- **Confusion Matrix**: To visualize true/false positives and negatives.
- **Precision, Recall, and F1-Score**: To measure the model's accuracy on the minority class.
- **ROC-AUC Score**: To evaluate the model's overall ability to distinguish between classes.

---

## 🚀 Results Summary

The evaluation of the trained models on the unseen test data was performed with a special focus on metrics suitable for imbalanced datasets. The code's output clearly shows that the **Random Forest classifier** significantly outperformed all other models, establishing itself as the most effective solution for this fraud detection problem.

A summary of the performance on the positive class (Fraud = 1), taken directly from the notebook's classification reports, is presented below:

| Model | Precision | Recall | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.29 | 0.26 | 0.27 | 0.61 |
| K-Nearest Neighbors | 0.14 | 0.28 | 0.19 | 0.58 |
| Decision Tree | 0.29 | 0.31 | 0.30 | 0.63 |
| **Random Forest** | **0.55** | 0.26 | **0.35** | **0.78** |

**Key Insights from the Results:**

- **Superior Performance of Random Forest**: This model achieved the highest **F1-Score (0.35)** and the highest **ROC-AUC Score (0.78)**. The F1-Score indicates the best balance between precision and recall, while the high ROC-AUC score confirms its strong capability in distinguishing between fraudulent and non-fraudulent cases.
- **Importance of Precision**: In a real-world fraud detection system, precision is critical to minimize the cost of investigating legitimate claims that are incorrectly flagged. The Random Forest model's precision of **0.55** is substantially higher than other models, making it a much more reliable choice for practical deployment.
- **Impact of SMOTE**: The successful training of the models, especially the Random Forest, was heavily reliant on applying the **SMOTE** technique to the training data. This step was crucial for enabling the models to learn the patterns of the minority (fraud) class effectively.

In conclusion, the combination of **SMOTE for data balancing** and the **Random Forest algorithm for classification** proved to be the most successful strategy, delivering a robust and accurate model for this task.

---

## 💻 Technologies Used

- Python 3
- Pandas & NumPy
- Scikit-learn
- imblearn (for SMOTE)
- Matplotlib & Seaborn
- Jupyter Notebook

---

## ⚙️ How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/arsalankhd/Vehicle-Claim-Fraud-Detection.git](https://github.com/arsalankhd/Vehicle-Claim-Fraud-Detection.git)
    ```
2.  Navigate to the directory:
    ```bash
    cd Vehicle-Claim-Fraud-Detection
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyterlab
    ```
4.  Open the `fraud_detection_model.ipynb` file in Jupyter.
