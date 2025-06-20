# Rainfall Prediction Model For THE DAILY BUZZ 🌧️
## _Predicting Tomorrow's Weather with Machine Learning Magic_ ✨

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/) [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-yellow?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)   [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)   [![Numpy](https://img.shields.io/badge/Numpy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy)](https://numpy.org/)   [![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/)   [![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Graphics-cyan?style=for-the-badge&logo=seaborn)](https://seaborn.pydata.org/)

#### "_Will it rain tomorrow?_" - A question as old as civilization, now answered with 84% accuracy using advanced machine learning! 🎯

---

## 🌟 Project Highlights
This project tackles the age-old challenge of weather prediction by building a robust machine learning system that forecasts rainfall with impressive accuracy. Using comprehensive meteorological data from Sydney, Australia, we've developed a **classification model** that can predict whether it will rain tomorrow based on today's weather conditions.

---

## 🎯 Problem Statement
**The Daily Buzz**, a Sydney-based newspaper company, needed a reliable system to predict rainfall for their weather section. This project addresses that need by creating a binary classification model that predicts whether it will rain tomorrow (Yes/No) based on comprehensive weather features.

---

## 🧰 Dataset

- Dataset: `sydney_rain prediction.xlsx`  
- Includes weather features such as:
  - Rainfall, Temperature, Wind, Humidity, Pressure, Evaporation
  - Target variable: **RainTomorrow (Yes/No)**

---

## 📊 Dataset Overview
Our model learns from rich meteorological data including:

| Feature Category | Variables | Description |
|------------------|-----------|-------------|
| **Temperature** | MinTemp, MaxTemp, Temp9am, Temp3pm | Daily temperature readings |
| **Humidity** | Humidity9am, Humidity3pm | Atmospheric moisture levels |
| **Pressure** | Pressure9am, Pressure3pm | Barometric pressure measurements |
| **Weather Conditions** | Rainfall, Evaporation, Sunshine | Precipitation and solar data |
| **Cloud Cover** | Cloud9am, Cloud3pm | Sky conditions throughout the day |
| **Target Variable** | RainTomorrow | Binary outcome (Yes/No) |

---

## 🔄 Data Preprocessing

- Loaded dataset using `pandas`  
- Performed initial exploratory data analysis: `.info()`, `.describe()`, `.head()`  
- Handled missing values appropriately  
- Applied **StandardScaler** for feature scaling  
- Converted categorical variables into numerical format for model compatibility

---

## 🤖 Machine Learning Models

Multiple classification models were implemented and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Naive Bayes Classifier  
- Gradient Boosting Classifier


---

## 📊 Evaluation Metrics

Each model was evaluated based on:

- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1-Score  
- ROC AUC Curve (if applicable)

---

## 📊 Evaluation Results

| Model                  | Accuracy | Precision | Recall | F1‑Score | ROC AUC |
| ---------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression    | 82.47 %  | 0.820     | 0.830  | 0.825    | 0.905   |
| K‑Nearest Neighbors    | 78.95 %  | 0.785     | 0.790  | 0.787    | 0.867   |
| Decision Tree          | 80.12 %  | 0.802     | 0.800  | 0.801    | 0.872   |
| **Random Forest**      | 86.33 %  | 0.863     | 0.860  | 0.861    | 0.925   |
| Support Vector Machine | 85.15 %  | 0.852     | 0.855  | 0.853    | 0.918   |
| Naive Bayes            | 79.44 %  | 0.794     | 0.794  | 0.794    | 0.860   |
| Gradient Boosting      | 87.02 %  | 0.870     | 0.870  | 0.870    | 0.932   |

---

## ✅ Best Model: Gradient Boosting
**Gradient Boosting** leads in accuracy **(87.02%—highest overall)** and also displays strong precision, recall, F1-score, and ROC AUC **(0.932)**, making it the top-performing classifier in our project.

---

## 🗂 Repository Structure

```
├── Rainfall_Prediction_Project.ipynb # Complete notebook
├── sydney_rain prediction.xlsx # Input dataset
├── requirements.txt
└── README.md
```
---

## 🛠️ Technical Implementation

Prerequisites
- ```
  pip install numpy
- ```
  pip install pandas
- ```
  pip install matplotlib
- ```
  pip install seaborn
- ```
  pip install scikit-learn


### 🎯 How to Run 

1️⃣ Clone this repository

```bash
git clone https://github.com/yourusername/Rainfall-Prediction-Project.git
cd Rainfall-Prediction-Project
```

2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

3️⃣ Launch the notebook

```bash
jupyter notebook Rainfall_Prediction_Project.ipynb
```
---

## 🔭 Future Improvements
- Hyperparameter tuning for each classifier
- Model ensemble or stacking
- Incorporate external weather APIs for real-time prediction
- Deploy as a web application for public use

### *Feature Importance Analysis*
Our model identifies the most influential weather patterns:

- Humidity levels (9am & 3pm readings)
- Pressure variations (atmospheric pressure changes)
- Temperature ranges (daily min/max temperatures)
- Cloud cover patterns (morning vs afternoon)

---

## 🤝 Contributing
We welcome contributions! Whether it's:

- 🐛 Bug fixes
- 📊 New feature engineering ideas
- 🔬 Alternative model implementations
- 📚 Documentation improvements

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎉 Acknowledgments

- The Daily Buzz for providing the business context
- Sydney Weather Bureau for comprehensive meteorological data
- Scikit-learn community for excellent ML tools
- Open source contributors making data science accessible

---

#### 🌟 Star this repo if you found it helpful!
#### _Built with ❤️ and lots of ☕ by a data science enthusiast_
#### **Want to predict tomorrow's weather? Clone, run, and let the algorithms do the magic!✨**

---

<p align="center">Thank you!!!</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/02d0eecb-7d96-4970-b03a-f284089287ed" alt="Sticker Image">
</p>

<p align="center">...</p>

<p align="center"> Have a cup of coffee ;)
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d94be366-eeb1-4de5-86a4-bf6a61224a83"alt="Sticker Image">
</p>

