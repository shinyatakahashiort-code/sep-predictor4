# 👁️ Spherical Equivalent (SE) Prediction App

A multilingual web application for predicting spherical equivalent (SE) from ophthalmological examination data using multiple machine learning models.

眼科検査データから球面度数を予測する多言語対応Webアプリケーション

## 🌟 Features

- **13 Machine Learning Models**: Compare predictions from Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVR, KNN, and MLP
- **Multilingual Support**: Switch between English and Japanese (日本語)
- **Interactive Interface**: Easy-to-use sidebar for input parameters
- **Real-time Predictions**: Instant predictions from all models
- **Performance Metrics**: View Test R², RMSE, and CV RMSE for each model
- **Visual Comparisons**: Bar charts comparing predictions across models

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/se-prediction-app.git
cd se-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file path to `app.py`
6. Click "Deploy"!

## 📊 Model Performance

| Model | Test R² | Test RMSE | CV RMSE |
|-------|---------|-----------|---------|
| **Gradient Boosting** | 0.9640 | 0.7369 | 0.9030 |
| Extra Trees | 0.9609 | 0.7674 | 0.9017 |
| MLP | 0.9601 | 0.7753 | 1.4036 |
| XGBoost | 0.9562 | 0.8123 | 0.9912 |
| Ridge | 0.9547 | 0.8261 | 0.9182 |

## 🔧 Training Your Own Models

If you want to train models with your own data:

```python
# 1. Prepare your data (CSV with required columns)
import pandas as pd
sep_df_cleaned = pd.read_csv('your_data.csv')

# Required columns: 年齢, 性別, K（AVG）, AL, LT, ACD, SE_p

# 2. Run the training script
python train_and_save_models.py

# 3. Models will be saved in the models/ directory
```

## 📁 Project Structure

```
se-prediction-app/
├── app.py                      # Main Streamlit application
├── train_and_save_models.py    # Script to train and save models
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Directory for saved models (optional)
│   ├── scaler.pkl
│   ├── Gradient_Boosting.pkl
│   ├── Extra_Trees.pkl
│   └── ...
└── .gitignore                  # Git ignore file
```

## 🎯 Input Parameters

### Required Input Features

- **Age (年齢)**: Patient age (0-100 years)
- **Sex (性別)**: Patient sex (Male/Female)
- **K (AVG)**: Average Keratometry (40.0-50.0)
- **AL**: Axial Length (眼軸長, 20.0-30.0 mm)
- **LT**: Lens Thickness (水晶体厚, 3.0-6.0 mm)
- **ACD**: Anterior Chamber Depth (前房深度, 2.0-4.0 mm)

## 🌐 Language Support

The app supports:
- **English** - Full interface in English
- **日本語** - 完全日本語対応

Switch languages using the sidebar dropdown.

## 📈 Usage Examples

### Example 1: Single Patient Prediction
1. Select language (English/日本語)
2. Enter patient data in the sidebar
3. Click "Predict" button
4. View predictions from all models
5. Compare results in the chart

### Example 2: Model Comparison
1. Uncheck "Show All Models"
2. Select specific models to compare
3. Run prediction
4. Analyze differences between selected models

## 🛠️ Technical Details

- **Framework**: Streamlit 1.31.0
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Cross-Validation**: 5-Fold CV
- **Random Seed**: 2025 (for reproducibility)

## 📝 Model Descriptions

### Top Performing Models

1. **Gradient Boosting** (Best Overall)
   - Test R²: 0.9640
   - Balanced performance and stability
   - Recommended for production use

2. **Extra Trees**
   - Test R²: 0.9609
   - Fast predictions
   - Good generalization

3. **MLP (Neural Network)**
   - Test R²: 0.9601
   - Complex patterns recognition
   - Higher variance in CV

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Built with Streamlit
- Machine learning models from scikit-learn, XGBoost, LightGBM, and CatBoost
- Ophthalmological data analysis

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This app is for educational and research purposes only. Always consult with qualified healthcare professionals for medical decisions.

**注意**: このアプリは教育・研究目的のみです。医療上の判断については必ず専門医にご相談ください。
