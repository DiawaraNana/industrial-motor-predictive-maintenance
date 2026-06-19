import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Motor Health AI Diagnostic", page_icon="⚙️", layout="wide")

st.title("⚙️ Motor Diagnostics App")
st.markdown("Upload your vibration data (e.g., `valid.csv` or `test.csv`) to get instant AI predictions for all machine samples.")

# Cache the model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model_and_encoder():
    bst = lgb.Booster(model_file='motor_fault_model.txt')
    le = joblib.load('label_encoder.pkl')
    return bst, le

# Attempt to load the model immediately 
try:
    bst, le = load_model_and_encoder()
    st.success("✅ AI Model and Label Dictionary loaded successfully.")
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please ensure `motor_fault_model.txt` and `label_encoder.pkl` are in the same folder.")
    st.stop() # Stops the rest of the app from running until the model is found

# ==========================================
# FILE UPLOAD & BATCH PREDICTION
# ==========================================
uploaded_file = st.file_uploader("Upload your machine data CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the uploaded dataset
        new_data = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(new_data)} samples for analysis.")
        
        # Check if 'label' column exists (so we can compare AI answers to actual answers)
        if 'label' in new_data.columns:
            actuals = new_data['label'].values
            X_new = new_data.drop('label', axis=1)
            has_actuals = True
        else:
            X_new = new_data
            has_actuals = False
        
        # Make Predictions on the whole dataset
        with st.spinner("AI is analyzing all vibration signatures..."):
            predictions_prob = bst.predict(X_new)
            predicted_classes = np.argmax(predictions_prob, axis=1)
            confidence_scores = np.max(predictions_prob, axis=1) * 100
            human_labels = le.inverse_transform(predicted_classes)
        
        # Format the output for the screen
        results_dict = {
            "Sample ID": [f"Sample_{i+1}" for i in range(len(human_labels))],
            "AI Diagnosis": human_labels,
            "Confidence": [f"{score:.2f}%" for score in confidence_scores]
        }
        
        if has_actuals:
            results_dict["True Status"] = actuals
            # Add a visual marker column to easily see if the AI was correct
            results_dict["Match?"] = ["✅" if p == a else "❌" for p, a in zip(human_labels, actuals)]

        results_df = pd.DataFrame(results_dict)
        
        # --- Summary Metrics ---
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples Analyzed", len(results_df))
        
        faults_found = len(results_df[results_df['AI Diagnosis'] != 'healthy'])
        col2.metric("Total Faults Detected", faults_found)
        
        if has_actuals:
            accuracy = (results_df["Match?"] == "✅").mean() * 100
            col3.metric("AI Accuracy on this File", f"{accuracy:.1f}%")

        # --- Detailed Report Table ---
        st.subheader("Detailed Diagnostic Report")
        
        # Highlight any row where the AI detects a fault (not healthy)
        def highlight_faults(row):
            if row['AI Diagnosis'] != 'healthy':
                return ['background-color: #ffe6e6'] * len(row) # Light red background
            return [''] * len(row)
            
        st.dataframe(results_df, use_container_width=True)
        
        # Download the results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Report", csv, "batch_diagnostic_report.csv", "text/csv")

    except Exception as e:
         st.error(f"Error processing the file: {e}")