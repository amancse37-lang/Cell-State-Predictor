import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Page Configuration
st.set_page_config(page_title="CellFate-ML Dashboard", page_icon="🧬", layout="wide")

st.title("🧬 CellFate-ML: Cell Reprogramming State Predictor")
st.write("Upload a gene expression sample dataset to instantly predict cellular reprogramming states using your custom trained Neural Network.")

# Load the saved trained model artifacts
@st.cache_resource
def load_trained_model():
    model_path = 'cellfate_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

artifacts = load_trained_model()

if artifacts is None:
    st.error("⚠️ Model file ('cellfate_model.pkl') not found! Please train and save the model from your Jupyter Notebook first.")
else:
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    w1, b1 = artifacts['dense1_weights'], artifacts['dense1_biases']
    w2, b2 = artifacts['dense2_weights'], artifacts['dense2_biases']

    # File uploader
    uploaded_file = st.file_uploader("Upload Gene Expression CSV File", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(input_df.head())
        st.write(f"Dataset Shape: {input_df.shape[0]} samples, {input_df.shape[1]} features.")
        
        if st.button("Run Real-Time Prediction", type="primary"):
            with st.spinner("Running preprocessing, PCA, and Neural Network forward pass..."):
                try:
                    # 1. StandardScaler transformation
                    X_scaled = scaler.transform(input_df)
                    
                    # 2. PCA Dimensionality Reduction
                    X_pca = pca.transform(X_scaled)
                    
                    # 3. Custom Dense Layer 1 + ReLU
                    z1 = np.dot(X_pca, w1) + b1
                    a1 = np.maximum(0, z1)
                    
                    # 4. Custom Dense Layer 2 + Softmax
                    z2 = np.dot(a1, w2) + b2
                    exp_shifted = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                    y_pred_probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
                    
                    # Get predicted class indices
                    predicted_classes = np.argmax(y_pred_probs, axis=1)
                    
                    # Map class indices to biological cell states
                    cell_states = {
                        0: "Somatic / Skin Cell",
                        1: "Induced Pluripotent Stem Cell (iPSC)",
                        2: "Neuron",
                        3: "Progenitor Cell",
                        4: "Mature Tissue Cell"
                    }
                    
                    input_df['Predicted_Cell_State'] = [cell_states.get(p, "Unknown") for p in predicted_classes]
                    
                    st.success("Prediction Completed Using Your Trained Custom MLP!")
                    st.subheader("🎯 Final Prediction Results")
                    st.dataframe(input_df[['Predicted_Cell_State']])
                    
                    # Summary chart
                    st.bar_chart(input_df['Predicted_Cell_State'].value_counts())
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.info("Awaiting file upload. Please upload your test CSV file to run inference.")