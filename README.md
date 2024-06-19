# cancer_prediction_app
# Breast Cancer Detection Using Logistic Regression

## Overview

This project employs logistic regression to predict breast cancer based on the Wisconsin Diagnostic Dataset. The dataset comprises features describing cell characteristics from digitized images of breast mass fine needle aspirates. The primary objective is to classify tumors as benign or malignant, providing a tool for initial screening and diagnosis.

## Features

- **Clump Thickness**: Thickness of the cell clump (1-10)
- **Uniformity of Cell Size**: Consistency in cell sizes (1-10)
- **Uniformity of Cell Shape**: Consistency in cell shapes (1-10)
- **Marginal Adhesion**: Degree of cell adhesion (1-10)
- **Single Epithelial Cell Size**: Size of single epithelial cells (1-10)
- **Bare Nuclei**: Presence of bare nuclei (1-10)
- **Bland Chromatin**: Chromatin texture (1-10)
- **Normal Nucleoli**: Presence of normal nucleoli (1-10)
- **Mitoses**: Number of mitoses (1-10)

## Installation

To run the project, install the required dependencies using:

```bash
pip install -r requirements.txt
Usage
Launch the Streamlit app with:

Model Training
The logistic regression model is trained on the Wisconsin Diagnostic Dataset, following these steps:

Data Loading and Preprocessing: The dataset is loaded from the UCI Machine Learning Repository. Missing values are handled, and the data is cleaned.
Feature and Target Separation: The features are separated from the target variable (class label).
Train-Test Split: The data is split into training and testing sets.
Model Training: A logistic regression model is trained using the training data.
Evaluation Metrics
Confusion Matrix
The confusion matrix provides a detailed breakdown of the model's performance by comparing the actual and predicted classifications. This matrix is crucial for understanding the types of errors the model makes.

python
Copy code
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16}, linewidths=0.5)
ROC Curve
The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of the model by plotting the true positive rate against the false positive rate at various threshold settings. The area under the ROC curve (AUC) provides a single measure of overall model performance.

python
Copy code
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
Visualization
User Input Parameters
Users can adjust the features using sliders in the sidebar and see the prediction results immediately. The interface is designed for ease of use, allowing for interactive exploration of different feature combinations.

Radar Chart for Feature Comparison
A radar chart is used to visually compare the user input features against the average features of benign and malignant cases. This comparison helps in understanding how the input features align with typical benign and malignant profiles.

python
Copy code
fig_radar_input = radar_chart(input_features, 'User Input')
fig_radar_input.add_trace(go.Scatterpolar(r=mean_benign, theta=mean_benign.index, fill='toself', name='Mean Benign'))
fig_radar_input.add_trace(go.Scatterpolar(r=mean_malignant, theta=mean_malignant.index, fill='toself', name='Mean Malignant'))
Conclusion
This project demonstrates the application of logistic regression in predicting breast cancer based on cell characteristics. By leveraging interactive visualizations and evaluation metrics, users can gain a deeper understanding of the model's performance and the significance of various features.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
The UCI Machine Learning Repository for providing the dataset.
The Streamlit community for the excellent framework that made this interactive application possible.
The developers and contributors to the Python libraries used in this project: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Plotly.
For any inquiries or contributions, please feel free to open an issue or submit a pull request.
