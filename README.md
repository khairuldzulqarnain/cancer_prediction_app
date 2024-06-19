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

bash
Copy code
streamlit run app.py
Explore the sliders on the sidebar to adjust features and view predictions instantly.

Model Training
The logistic regression model is trained on the Wisconsin Diagnostic Dataset, focusing on data preprocessing and feature engineering to enhance prediction accuracy.

Evaluation Metrics
Confusion Matrix
The confusion matrix illustrates model performance by comparing actual versus predicted classifications, offering insights into prediction errors.

ROC Curve
The ROC curve demonstrates the model's diagnostic ability by plotting true positive rate against false positive rate, with the AUC providing a summary of overall performance.

Visualization
User Input Parameters
Adjust feature sliders in the sidebar to explore different combinations and observe immediate prediction outcomes.

Radar Chart for Feature Comparison
Compare user input features with average benign and malignant features using radar charts, aiding in understanding feature impacts on predictions.

Conclusion
This project showcases logistic regression’s application in breast cancer prediction, leveraging interactive visualizations and evaluation metrics to enhance understanding and decision-making.

License
This project is licensed under the MIT License. See LICENSE for details.

Acknowledgements
UCI Machine Learning Repository for providing the dataset.
Streamlit community for their excellent framework.
Contributors to Python libraries used in this project: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Plotly.
For inquiries or contributions, feel free to open an issue or submit a pull request.
