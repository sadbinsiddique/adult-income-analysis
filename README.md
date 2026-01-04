# `End-to-End` ML Project **_Income Prediction_**

[![Python](https://img.shields.io/badge/Python-%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![UV](https://img.shields.io/badge/uv-0.9.21-3776AB?logo=uv&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Jinja2](https://img.shields.io/badge/Jinja2-3.1-B41717?logo=jinja&logoColor=white)](https://jinja.palletsprojects.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-7.0-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

## Project Overview

This **end-to-end machine learning project** analyzes adult income data to predict whether an individual earns **more or less than $50K per year**.  
The project uses **Support Vector Machine**, provides **comprehensive data visualizations** and demonstrates a **production-ready API** for model predictions.

---

## Project Requirements

1) Load the dataset into your program using the Pandas library.
2) Apply appropriate data cleaning techniques to your dataset using the Pandas library. Replace bad data using suitable methods; do not delete any records except for duplicates.
3) Analyze the `frequency distributions` of the dataset’s features by generating plots with Matplotlib. Use the `plt.subplots()` method to display all charts within a single figure.
4) Perform scaling on the features of the dataset. You will need to apply data conversion before scaling if required.
5) Split your dataset into training and testing subsets using the `train_test_split()` function and set the random_state parameter to `3327`.
6) Apply Support Vector Machine `SVM` Classifier to the dataset. Build `train` your prediction model in this step.
7) Compute the confusion matrix for the SVM model. Provide an in-depth discussion in your report.
8) Calculate and compare the training accuracy and test accuracy of your model.
9) Evaluate an `SVM` model using a `10-fold cross-validation` strategy. Provide a detailed report on the model’s predictive accuracy and compare this result with the accuracy of the previous model trained in Task 6.

## Getting Started

### 1️. Clone the Repository

```cmd
git clone https://github.com/sadbinsiddique/adult-income-analysis.git
```

### 1.1 Go to the Repository

```cmd
cd adult-income-analysis
```

### 2. Create Virtual Environment

```cmd
uv venv --python=3.10
```

### 2.1 Activate Virtual Environment

```cmd
.venv\Scripts\activate
```

### 2.2 Install Dependencies

```cmd
uv pip install -r requirements.txt --link-mode=copy
```

### 3. Run Server `Backend` + `Frontend`

```cmd
python app.py
```

