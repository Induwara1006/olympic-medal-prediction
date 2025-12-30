# 🏅 Olympic Medal Prediction Model

A comprehensive machine learning project that predicts Olympic medal counts for teams using historical data, athlete demographics, and team characteristics. Built with Python and scikit-learn.

## 📊 Project Overview

This project implements multiple regression models to forecast Olympic medal performance. By analyzing team composition, athlete statistics, and historical performance, the models can predict medal counts with high accuracy, helping understand key factors that contribute to Olympic success.

## ✨ Key Features

- **Advanced Data Preprocessing**
  - Automatic handling of missing values
  - One-hot encoding for categorical variables (team, country)
  - Feature scaling using StandardScaler for optimal model performance

- **Multiple ML Algorithms**
  - Linear Regression baseline model
  - Random Forest Regressor for enhanced accuracy
  - Post-processing to ensure realistic predictions (non-negative, integer medals)

- **Comprehensive Model Evaluation**
  - R² Score (coefficient of determination)
  - Mean Absolute Error (MAE)
  - Threshold accuracy (predictions within ±5 medals)
  - Binary classification confusion matrix (medal vs. no medal)

- **Rich Visualizations**
  - Actual vs. Predicted scatter plots
  - Residual distribution analysis
  - Confusion matrix heatmaps
  - Model performance comparisons

- **Model Persistence**
  - Save trained models using joblib
  - Export scalers for production deployment
  - Easy model loading for future predictions

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Core Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms and tools
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical graphics
  - `joblib` - Model serialization

## 📁 Project Structure

```
olympic-medal-prediction/
│
├── Medal_Prediction.ipynb      # Main Jupyter notebook with complete workflow
├── teams.csv                    # Training dataset with team statistics
├── olympic_medal_predictor.pkl # Saved trained model (generated)
├── olympic_scaler.pkl          # Saved feature scaler (generated)
├── lr_performance.png          # Model performance visualization (generated)
└── README.md                   # Project documentation
```

## 📊 Dataset Description

The model uses [`teams.csv`](teams.csv) with the following features:

### Input Features
| Feature | Description | Type |
|---------|-------------|------|
| `events` | Number of events participated in | Numerical |
| `athletes` | Number of athletes in the team | Numerical |
| `age` | Average age of athletes | Numerical |
| `height` | Average height of athletes (cm) | Numerical |
| `weight` | Average weight of athletes (kg) | Numerical |
| `prev_medals` | Total medals won in previous Olympics | Numerical |
| `prev_3_medals` | Medals won in last 3 Olympic Games | Numerical |
| `team` | Team/organization name | Categorical |
| `country` | Country represented | Categorical |

### Target Variable
- **`medals`** - Number of medals won (to be predicted)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Induwara1006/olympic-medal-prediction.git
cd olympic-medal-prediction
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Launch Jupyter Notebook**
```bash
jupyter notebook Medal_Prediction.ipynb
```

2. **Execute cells sequentially** to:
   - Load and explore the dataset
   - Preprocess and encode features
   - Train multiple models
   - Evaluate and compare performance
   - Generate visualizations
   - Save trained models

## 📈 Model Performance

### Linear Regression Model
- **Purpose**: Baseline model for comparison
- **Metrics**:
  - R² Score: Measures proportion of variance explained
  - Mean Absolute Error: Average prediction error in medal count
  - Threshold Accuracy: Percentage of predictions within ±5 medals
- **Advantages**: Simple, interpretable, fast training

### Random Forest Regressor
- **Configuration**: 100 decision trees
- **Purpose**: Enhanced accuracy through ensemble learning
- **Metrics**:
  - Higher R² score compared to linear regression
  - Lower MAE for more accurate predictions
  - Better threshold accuracy
- **Advantages**: Handles non-linear relationships, reduces overfitting

### Model Outputs
Both models include:
- Predictions clipped at 0 (no negative medals)
- Rounded to nearest integer
- Comprehensive evaluation metrics
- Training vs. testing performance comparison

## 🎯 Usage Examples

### Training the Model

```python
# The notebook handles all training automatically
# Simply run all cells in sequence
```

### Making Predictions with Saved Model

```python
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('olympic_medal_predictor.pkl')
scaler = joblib.load('olympic_scaler.pkl')

# Prepare new data (must match training format)
new_data = pd.DataFrame({
    'events': [15],
    'athletes': [120],
    'age': [26.5],
    'height': [175.0],
    'weight': [72.0],
    'prev_medals': [25],
    'prev_3_medals': [18],
    # Include one-hot encoded team and country columns...
})

# Scale numerical features
num_cols = ['events', 'athletes', 'age', 'height', 'weight', 'prev_medals', 'prev_3_medals']
new_data[num_cols] = scaler.transform(new_data[num_cols])

# Make prediction
prediction = model.predict(new_data)
prediction = np.maximum(prediction, 0).round()

print(f"Predicted medals: {int(prediction[0])}")
```

## 📊 Visualization Gallery

The project generates several insightful visualizations:

1. **Actual vs. Predicted Medals** - Scatter plot showing model accuracy
2. **Residual Distribution** - Histogram of prediction errors
3. **Confusion Matrix** - Binary classification (medal/no medal) performance
4. **Training vs. Testing Comparison** - Overfitting analysis

## 🔍 Key Insights

- **Feature Importance**: Previous medal counts are strong predictors
- **Scaling Impact**: StandardScaler significantly improves model convergence
- **Model Comparison**: Random Forest typically outperforms Linear Regression
- **Threshold Accuracy**: Most predictions fall within ±5 medals of actual values

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add some amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Ideas
- Add more ML algorithms (XGBoost, Neural Networks)
- Implement hyperparameter tuning
- Add feature engineering techniques
- Create a web interface for predictions
- Expand dataset with more historical data

## 📝 Future Enhancements

- [ ] Implement cross-validation for robust evaluation
- [ ] Add hyperparameter optimization (GridSearchCV)
- [ ] Include feature importance visualization
- [ ] Create an interactive dashboard (Streamlit/Plotly)
- [ ] Deploy model as REST API
- [ ] Add support for real-time predictions
- [ ] Implement deep learning models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Induwara Abhisheka**
- GitHub: [@Induwara1006](https://github.com/Induwara1006)
- Email: induwaraabhisheka99@gmail.com

## 🙏 Acknowledgments

- Olympic historical data contributors
- scikit-learn documentation and community
- Open source machine learning community

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Olympic Games Dataset Information](https://www.olympic.org/)

---

**⭐ If you found this project helpful, please consider giving it a star!**

*Last Updated: December 2025*
