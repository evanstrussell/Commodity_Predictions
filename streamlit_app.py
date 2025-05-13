import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Commodity Classifier", layout="wide")

@st.cache_data
def load_excel(file):
    ref = pd.read_excel(file, sheet_name="Last Price Paid", usecols=["DESCRIPTION", "New Commodity"])
    mod = pd.read_excel(file, sheet_name="Grainger", usecols=["Material Description", "Commodity"])
    return ref, mod

def preprocess(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"[^a-z0-9, ]", " ", text.lower())

def train_model(df, text_col, target_col, vectorizer=None):
    X_text = df[text_col].apply(preprocess)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=2000, min_df=2)
        X = vectorizer.fit_transform(X_text)
    else:
        X = vectorizer.transform(X_text)
    y = df[target_col]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf, vectorizer

def get_confidences(clf, X):
    probs = clf.predict_proba(X)
    preds = clf.classes_[np.argmax(probs, axis=1)]
    confs = np.max(probs, axis=1)
    return preds, confs

# Session state init
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = pd.DataFrame()
    st.session_state.prev_all_correct = False
    st.session_state.stop_training = False
    st.session_state.iteration = 0

st.title("🔁 Human-in-the-Loop Commodity Classifier")

uploaded_file = st.file_uploader("Upload the Savings Tracker Excel File", type=["xlsx"])

if uploaded_file:
    reference_df, model_df = load_excel(uploaded_file)

    # Clean and prep
    reference_df['clean_desc'] = reference_df['DESCRIPTION'].apply(preprocess)
    model_df['clean_desc'] = model_df['Material Description'].apply(preprocess)

    # Train model
    model, vec = train_model(reference_df, "clean_desc", "New Commodity")

    # Filter Grainger
    model_df_not_found = model_df[model_df['Commodity'].str.lower() == 'not found'].copy()
    model_df_not_found = model_df_not_found[model_df_not_found['clean_desc'].str.strip() != ""]
    X_not_found = vec.transform(model_df_not_found['clean_desc'])

    # Predict and get confidences
    preds, confs = get_confidences(model, X_not_found)
    model_df_not_found['Predicted Commodity'] = preds
    model_df_not_found['Confidence'] = confs

    threshold = model_df_not_found["Confidence"].quantile(0.25)
    low_conf_df = model_df_not_found[model_df_not_found["Confidence"] <= threshold]

    sample = low_conf_df.sample(n=15, random_state=st.session_state.iteration)
    #sample = model_df_not_found.sample(n=15, random_state=st.session_state.iteration)

    st.subheader("🔍 Review Sample Predictions")

    with st.form("corrections_form"):
        updated_rows = []
        all_correct = True

        for i, row in sample.iterrows():
            st.markdown(f"**Row {i+1}:**")
            st.write(f"**Description:** {row['Material Description']}")
            st.write(f"**Predicted:** {row['Predicted Commodity']} | **Confidence:** {round(row['Confidence']*100, 2)}%")

            override = st.selectbox(
                f"Confirm or correct the commodity for Row {i+1}",
                options=[row['Predicted Commodity']] + sorted(reference_df['New Commodity'].dropna().unique()),
                key=f"select_{i}"
            )

            corrected = override != row['Predicted Commodity']
            if corrected:
                all_correct = False

            updated_rows.append({
                'Material Description': row['Material Description'],
                'Original Prediction': row['Predicted Commodity'],
                'Corrected Commodity': override,
                'Confidence': row['Confidence']
            })

        # Submit button *inside* the form
        submitted = st.form_submit_button("✅ Submit Corrections & Retrain")

    if submitted:
        new_feedback = pd.DataFrame(updated_rows)
        st.session_state.feedback_log = pd.concat([st.session_state.feedback_log, new_feedback], ignore_index=True)

        # Add corrections to reference_df
        new_training = new_feedback[['Material Description', 'Corrected Commodity']].rename(columns={
            'Material Description': 'DESCRIPTION',
            'Corrected Commodity': 'New Commodity'
        })
        new_training['clean_desc'] = new_training['DESCRIPTION'].apply(preprocess)
        reference_df = pd.concat([reference_df, new_training], ignore_index=True)

        # Update loop state
        if all_correct and st.session_state.prev_all_correct:
            st.session_state.stop_training = True
        st.session_state.prev_all_correct = all_correct
        st.session_state.iteration += 1

        st.rerun()

    if st.button("🛑 End Training Early"):
        st.session_state.stop_training = True
        st.rerun()

    if st.session_state.stop_training:
        st.success("✅ No corrections for two rounds — model loop complete!")

        # Final model prediction for all Not Found
        final_model, final_vec = train_model(reference_df, "clean_desc", "New Commodity")
        final_preds, final_confs = get_confidences(final_model, final_vec.transform(model_df_not_found['clean_desc']))

        model_df_not_found['Final Prediction'] = final_preds
        model_df_not_found['Final Confidence'] = final_confs

        # Save files
        st.subheader("📁 Download Outputs")

        st.download_button("📥 Original Predictions", data=model_df_not_found.to_csv(index=False), file_name="original_predictions.csv")
        st.download_button("📥 Manual Corrections", data=st.session_state.feedback_log.to_csv(index=False), file_name="manual_corrections.csv")

        final_updated = model_df_not_found[['Material Description', 'Final Prediction', 'Final Confidence']]
        st.download_button("📥 Final Updated Predictions", data=final_updated.to_csv(index=False), file_name="final_predictions.csv")