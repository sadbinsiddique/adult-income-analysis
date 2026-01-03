# End-to-End Adult Income Analysis

[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Jinja2](https://img.shields.io/badge/Jinja2-B41717?logo=jinja&logoColor=white)](https://jinja.palletsprojects.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-5C4EE5?logo=uvicorn&logoColor=white)](https://www.uvicorn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## Project Overview
This **end-to-end machine learning project** analyzes adult income data to predict whether an individual earns **more or less than $50K per year**.  
The project uses **gradient boosting algorithms**, provides **comprehensive data visualizations**, and demonstrates a **production-ready API** for model predictions.

---

## Project Features
- **Data Preprocessing:** Handling missing values, encoding categorical variables, feature scaling  
- **Exploratory Data Analysis:** Visualizations, correlation analysis, feature importance  
- **Machine Learning:** XGBoost, LightGBM, CatBoost for accurate predictions  
- **Model Evaluation:** Accuracy, F1-score, Confusion Matrix, ROC-AUC  
- **Backend API:** FastAPI + Uvicorn  
- **Web Interface:** Jinja2 templating for interactive prediction  
- **Notebook Exploration:** Jupyter Notebook for experiments

---

## Tools & Libraries
- [![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![UV](https://img.shields.io/badge/uv-0.9.21-3776AB?logo=uv&logoColor=white)](https://www.python.org/)
- [![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
	[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
	[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
	[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
- 
	[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
	[![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28?logo=xgboost&logoColor=white)](https://xgboost.ai/)
	[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
	[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
-  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-5C4EE5?logo=python&logoColor=white)](https://www.uvicorn.org/)  
- [![Jinja2](https://img.shields.io/badge/Jinja2-B41717?logo=jinja&logoColor=white)](https://jinja.palletsprojects.com/)
-  [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)



##  Getting Started

### 1️. Clone the Repository
```bash
git clone https://github.com/sadbinsiddique/adult-income-analysis.git
```

### 1.1 Go to the Repository

```bash
cd adult-income-analysis
```

### 2. Create Virtual Eiviroment 

```bash
uv venv --python=3.8
```

### 2.1 Active Virtual Eiviroment 

```bash
venv\Scripts\activate
```

### 2.2 Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Run Server `Backend` + `Frontend`

```bash
python app.py
```