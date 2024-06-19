import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go

# Load the dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    # Define column names based on the dataset description
    column_names = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 
                    'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 
                    'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    data = pd.read_csv(url, names=column_names)
    # Preprocess the data
    data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
    data['bare_nuclei'] = pd.to_numeric(data['bare_nuclei'])  # Convert 'bare_nuclei' to numeric
    data.dropna(inplace=True)  # Drop rows with missing values
    # Convert class label to binary (0: benign, 1: malignant)
    data['class'] = data['class'].map({2: 'Benign', 4: 'Malignant'})
    return data

# Load dataset
data = load_data()

# Split data into X (features) and y (target)
X = data.drop(['id', 'class'], axis=1)
y = data['class']

# Convert class labels to binary (0 and 1)
y_binary = y.map({'Benign': 0, 'Malignant': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Streamlit app
st.title('Breast Cancer Detection Apps 🎗️')

# Project Description
st.markdown("""
This project uses logistic regression to predict breast cancer based on the Wisconsin Diagnostic Dataset. The dataset consists of various features that describe cell characteristics derived from digitized images of fine needle aspirate of breast mass. 

The goal is to classify tumors as benign or malignant based on these features, providing a tool for preliminary screening and diagnosis.

### User Input Parameters 📊
Use the sliders on the left to adjust the features and see the prediction:

- **Clump Thickness**: Thickness of the clump (1-10)
- **Uniformity of Cell Size**: Uniformity of size of cells (1-10)
- **Uniformity of Cell Shape**: Uniformity of shape of cells (1-10)
- **Marginal Adhesion**: Degree of adhesion (1-10)
- **Single Epithelial Cell Size**: Size of epithelial cells (1-10)
- **Bare Nuclei**: Bare nuclei (1-10)
- **Bland Chromatin**: Chromatin (1-10)
- **Normal Nucleoli**: Normal nucleoli (1-10)
- **Mitoses**: Mitoses (1-10)

""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    features = {}
    features['clump_thickness'] = st.sidebar.slider('Clump Thickness 🌰', 1, 10, 5)
    features['cell_size_uniformity'] = st.sidebar.slider('Uniformity of Cell Size 🧬', 1, 10, 5)
    features['cell_shape_uniformity'] = st.sidebar.slider('Uniformity of Cell Shape 🔷', 1, 10, 5)
    features['marginal_adhesion'] = st.sidebar.slider('Marginal Adhesion 📌', 1, 10, 5)
    features['single_epithelial_cell_size'] = st.sidebar.slider('Single Epithelial Cell Size 🦠', 1, 10, 5)
    features['bare_nuclei'] = st.sidebar.slider('Bare Nuclei 🧫', 1, 10, 5)
    features['bland_chromatin'] = st.sidebar.slider('Bland Chromatin 🔬', 1, 10, 5)
    features['normal_nucleoli'] = st.sidebar.slider('Normal Nucleoli 🧽', 1, 10, 5)
    features['mitoses'] = st.sidebar.slider('Mitoses 🧬', 1, 10, 5)
    return pd.DataFrame([features])

input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters 📊')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display prediction
st.subheader('Prediction 📈')
if prediction[0] == 0:
    st.write('Benign')
else:
    st.write('Malignant')

# Display prediction probabilities
st.subheader('Prediction Probability 📉')
st.write(f'Probability of Benign: {prediction_proba[0][0]*100:.2f}%')
st.write(f'Probability of Malignant: {prediction_proba[0][1]*100:.2f}%')

# Display model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy * 100:.2f}%')

# Visualization: Confusion Matrix
st.subheader('Confusion Matrix 📊')
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16}, linewidths=0.5, ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')

# Display confusion matrix using st.pyplot() with explicit figure object
st.pyplot(fig_cm)

# Visualization: ROC Curve
st.subheader('ROC Curve 📈')
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic (ROC)')
ax_roc.legend(loc='lower right')

# Display ROC curve using st.pyplot() with explicit figure object
st.pyplot(fig_roc)

# Visualization: Radar Chart for Feature Comparison
st.subheader('Radar Chart: Feature Comparison 📊')

# Define function to generate radar chart
def radar_chart(features, title):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=features,
          theta=features.index,
          fill='toself',
          name=title
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 10]
        )),
      showlegend=True
    )

    return fig

# Get mean values for benign and malignant classes
mean_benign = X[y == 'Benign'].mean()
mean_malignant = X[y == 'Malignant'].mean()

# Plot radar charts based on user input and mean values
input_features = input_df.iloc[0]
fig_radar_input = radar_chart(input_features, 'User Input')
fig_radar_benign = radar_chart(mean_benign, 'Benign')
fig_radar_malignant = radar_chart(mean_malignant, 'Malignant')

# Display radar charts using st.plotly_chart()
st.plotly_chart(fig_radar_input)

# Add traces for benign and malignant
fig_radar_input.add_trace(go.Scatterpolar(
    r=mean_benign,
    theta=mean_benign.index,
    fill='toself',
    name='Mean Benign'
))

fig_radar_input.add_trace(go.Scatterpolar(
    r=mean_malignant,
    theta=mean_malignant.index,
    fill='toself',
    name='Mean Malignant'
))

# Update layout to show legend and adjust axes
fig_radar_input.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title='Radar Chart: Feature Comparison'
)

# Display updated radar chart
st.plotly_chart(fig_radar_input)

# Optionally, you can also display other visualizations like feature importance, etc., based on your model analysis.
