# Student Performance Index Analysis

A comprehensive data mining project analyzing factors that influence student exam performance using machine learning techniques.

## 📋 Project Overview

This project implements a complete data mining pipeline to predict and analyze student exam scores based on various demographic, academic, and lifestyle factors. The analysis includes both supervised learning (regression) and unsupervised learning (clustering) approaches.

## 🎯 Objectives

- Identify key factors influencing student exam performance
- Develop and compare multiple regression models to predict exam scores
- Perform clustering analysis to identify distinct student groups
- Extract actionable insights for educational interventions

## 📊 Dataset

The project uses the `StudentPerformanceFactors.csv` dataset containing:
- **6,607 student records**
- **20 features** including:
  - Academic factors: Hours Studied, Attendance, Previous Scores, Tutoring Sessions
  - Lifestyle factors: Sleep Hours, Physical Activity
  - Demographic factors: Gender, Family Income, Parental Education Level
  - Environmental factors: School Type, Teacher Quality, Distance from Home
  - Support factors: Parental Involvement, Access to Resources, Internet Access
  - Other factors: Extracurricular Activities, Motivation Level, Peer Influence, Learning Disabilities

## 🔧 Technologies Used

- **Python 3.x**
- **Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning models and preprocessing
  - `matplotlib` & `seaborn` - Data visualization
  - `jupyter` - Interactive notebook environment

## 📁 Project Structure

```
Milestone 3/
│
├── Milestone_3.ipynb              # Main Jupyter notebook with complete analysis
├── StudentPerformanceFactors.csv  # Dataset
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Project

1. Clone this repository:
```bash
git clone https://github.com/ParidhiBhardwajj/Student-Performance-Index.git
cd Student-Performance-Index
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Milestone_3.ipynb
```

3. Run all cells to execute the complete analysis pipeline.

## 📈 Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Imputed missing values in `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` using mode imputation
- **Encoding**: Applied one-hot encoding to categorical variables
- **Scaling**: Standardized numerical features using StandardScaler
- **Target Transformation**: Applied log transformation to handle skewness in the target variable

### 2. Supervised Learning Models
The project implements and compares four regression models:

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - L2 regularized regression with hyperparameter tuning via GridSearchCV
3. **Random Forest Regressor** - Ensemble method with feature importance analysis
4. **Support Vector Regression (SVR)** - Non-linear regression with RBF kernel

**Evaluation Metrics:**
- R² Score (Coefficient of Determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### 3. Unsupervised Learning
- **K-Means Clustering**: Identified optimal number of clusters using elbow method and silhouette score
- **Cluster Analysis**: Analyzed student groups based on performance patterns

### 4. Model Comparison
Comprehensive comparison of all models to identify the best performing approach for predicting exam scores.

## 🔍 Key Findings

- **Model Performance**: Multiple regression models were evaluated, with the best model achieving high predictive accuracy
- **Feature Importance**: Random Forest analysis identified the most critical factors affecting student performance
- **Student Clusters**: K-Means clustering revealed distinct student groups with different performance characteristics
- **Hyperparameter Optimization**: GridSearchCV successfully optimized model parameters

## 📊 Results

The notebook includes:
- Detailed model performance comparisons
- Feature importance visualizations
- Clustering analysis results
- Distribution plots and statistical summaries
- Comprehensive evaluation metrics for all models

## 🎓 Course Information

**Course**: CS 482 - Data Mining  
**Milestone**: 3  
**Focus**: Model Development and Experimentation

## 👤 Author

**Paridhi Bhardwaj**

## 📝 License

This project is part of an academic coursework assignment.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📧 Contact

For inquiries about this project, please reach out through GitHub.

---

**Note**: This project is for educational purposes as part of CS 482 coursework.

