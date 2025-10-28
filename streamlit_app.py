"""
streamlit_app.py
Auto-generated Streamlit dashboard from 'crime_dataset_india.csv'.
Features:
 - Loads the provided fixed dataset (path: /mnt/data/crime_dataset_india.csv)
 - Preprocesses data and trains a RandomForestClassifier to predict 'Case Closed' (Yes/No)
 - Provides interactive Plotly charts and a Folium map for top cities
 - Polished layout with sidebar controls and sections
Requirements (add these to requirements.txt):
streamlit
pandas
numpy
scikit-learn
plotly
folium
streamlit-folium
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib

DATA_PATH = Path("crime_dataset_india.csv")


st.set_page_config(page_title="Crime Prediction & Analysis", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(show_spinner=False)
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    # Parse dates
    for col in ["Date Reported", "Date of Occurrence", "Date Case Closed"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    # Time column to hour
    if "Time of Occurrence" in df.columns:
        # Some times like '05:00' or '14:46'
        df["Occ_Hour"] = pd.to_datetime(df["Time of Occurrence"], format="%H:%M", errors="coerce").dt.hour
    else:
        df["Occ_Hour"] = np.nan
    # Victim age to numeric
    if "Victim Age" in df.columns:
        df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    # Binary target: Case Closed -> 1 for Yes, 0 for No
    df["Case_Closed_Flag"] = df.get("Case Closed", "").apply(lambda x: 1 if str(x).strip().lower()=="yes" else 0)
    return df

@st.cache_data(show_spinner=False)
def train_model(df, random_state=42):
    # Select features
    features = []
    # Numeric features
    if "Victim Age" in df.columns:
        features.append("Victim Age")
    if "Police Deployed" in df.columns:
        features.append("Police Deployed")
    if "Occ_Hour" in df.columns:
        features.append("Occ_Hour")
    # Categorical features we'll one-hot encode: Victim Gender, Weapon Used, Crime Domain
    cat_cols = [c for c in ["Victim Gender", "Weapon Used", "Crime Domain"] if c in df.columns]
    X_num = df[features].fillna(-1)
    X_cat = df[cat_cols].fillna("Unknown")
    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    if len(cat_cols)>0:
        X_cat_enc = pd.DataFrame(encoder.fit_transform(X_cat), index=X_cat.index, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)
    else:
        X = X_num
        encoder = None
    y = df["Case_Closed_Flag"].fillna(0).astype(int)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)
    # Model
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    # Save artifacts
    artifacts = {"model": model, "encoder": encoder, "features": X.columns.tolist()}
    return artifacts, acc, report, cm

# A small lookup for lat/lon of major Indian cities in the dataset (extendable)
CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714),
    "Chennai": (13.0827, 80.2707),
    "Ludhiana": (30.9000, 75.8573),
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873)
}

def main():
    st.sidebar.title("Controls")
    st.sidebar.markdown("Use the controls to adjust the analysis and model training. The app uses the fixed dataset included with the notebook.")
    retrain = st.sidebar.button("Retrain model (takes a few seconds)")
    st.sidebar.markdown("### Filters")
    city_filter = st.sidebar.multiselect("Select cities to include (top cities available)", options=sorted(list(CITY_COORDS.keys())), default=[])

    st.title("Crime Prediction & Analysis Dashboard")
    st.markdown("This dashboard loads a fixed dataset and trains a model to predict whether a case will be closed (`Case Closed: Yes/No`).\n\n"
                "It also provides interactive visualizations and a map showing incidents for top cities (where coordinates are available).")

    # Load and preprocess data
    df = load_data()
    df = preprocess(df)

    # Apply city filter if selected
    if city_filter:
        df = df[df["City"].isin(city_filter)]

    # Overview
    st.header("Dataset overview")
    left, right = st.columns([1,1])
    with left:
        st.subheader("Snapshot")
        st.dataframe(df.head(10))
    with right:
        st.subheader("Quick stats")
        st.metric("Total records", df.shape[0])
        st.metric("Unique cities", int(df['City'].nunique() if 'City' in df.columns else 0))
        if 'Crime Description' in df.columns:
            st.metric("Unique crime types", int(df['Crime Description'].nunique()))

    # Interactive crime counts
    st.header("Exploratory Visualizations")
    if 'Crime Description' in df.columns:
        top_n = st.slider("Top N crime types to show", min_value=5, max_value=30, value=10)
        crime_counts = df['Crime Description'].value_counts().nlargest(top_n).reset_index()
        crime_counts.columns = ['Crime Description', 'Count']
        fig = px.bar(crime_counts, x='Crime Description', y='Count', title=f"Top {top_n} Crime Descriptions", hover_data=['Count'])
        fig.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Map for top cities
    st.header("Map — Top Cities (if coordinates available)")
    if 'City' in df.columns:
        city_counts = df['City'].value_counts().nlargest(20).reset_index()
        city_counts.columns = ['City', 'Count']
        # Keep only cities with known coords
        city_counts = city_counts[city_counts['City'].isin(CITY_COORDS.keys())]
        if not city_counts.empty:
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
            for _, row in city_counts.iterrows():
                c = row['City']
                cnt = int(row['Count'])
                lat, lon = CITY_COORDS[c]
                folium.CircleMarker(location=[lat, lon],
                                    radius=4 + np.log1p(cnt),
                                    popup=f"{c}: {cnt} incidents",
                                    color='crimson', fill=True).add_to(m)
            st_folium(m, width=700, height=450)
        else:
            st.info("No cities with known coordinates found in the top 20 cities. You can extend CITY_COORDS in the app source to add more cities.")

    # Model section
    st.header("Machine Learning — Predict Case Closed (Yes/No)")
    st.markdown("We train a RandomForest classifier using structured features (victim age, gender, weapon used, police deployed, occurrence hour).")

    # Train model (cached) or retrain on button press
    training_placeholder = st.empty()
    if retrain:
        with st.spinner("Retraining model..."):
            artifacts, acc, report, cm = train_model(df)
            # Save model
            joblib.dump(artifacts, "/mnt/data/crime_model_artifacts.pkl")
    else:
        with st.spinner("Training model (cached) if not already trained)..."):
            artifacts, acc, report, cm = train_model(df)

    st.subheader("Model performance on held-out test set")
    st.metric("Accuracy", f"{acc:.3f}")
    # Show confusion matrix
    cm_df = pd.DataFrame(cm, index=["Actual_No","Actual_Yes"], columns=["Pred_No","Pred_Yes"])
    st.write("Confusion matrix:")
    st.dataframe(cm_df)

    st.subheader("Classification report (summary)")
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df)

    # Prediction UI
    st.header("Make a prediction (single case)")
    st.markdown("Provide features for a single case to get a prediction (probability that the case will be closed).")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Victim Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Victim Gender", options=["M","F","Unknown"])
        with col2:
            weapon = st.selectbox("Weapon Used", options=sorted(df['Weapon Used'].fillna("Unknown").unique().tolist()))
            police_deployed = st.number_input("Police Deployed", min_value=0, max_value=1000, value=5)
        with col3:
            occ_hour = st.slider("Occurrence hour (0-23)", min_value=0, max_value=23, value=12)
            crime_domain = st.selectbox("Crime Domain", options=sorted(df['Crime Domain'].fillna("Unknown").unique().tolist()))
        submit = st.form_submit_button("Predict")

    if submit:
        # Build feature vector using artifacts info
        model = artifacts["model"]
        encoder = artifacts["encoder"]
        features = artifacts["features"]
        # Numeric vector
        x_num = pd.DataFrame([[age, police_deployed, occ_hour]], columns=[c for c in features if c in ["Victim Age","Police Deployed","Occ_Hour"]])
        # Categorical
        cat_cols = [c for c in ["Victim Gender","Weapon Used","Crime Domain"] if c in df.columns]
        x_cat = pd.DataFrame([[gender if "Victim Gender" in cat_cols else None,
                               weapon if "Weapon Used" in cat_cols else None,
                               crime_domain if "Crime Domain" in cat_cols else None]], columns=cat_cols)
        if encoder is not None and len(cat_cols)>0:
            x_cat_enc = pd.DataFrame(encoder.transform(x_cat.fillna("Unknown")), columns=encoder.get_feature_names_out(cat_cols))
            X_pred = pd.concat([x_num.reset_index(drop=True), x_cat_enc.reset_index(drop=True)], axis=1)
        else:
            X_pred = x_num
        # Align columns
        for c in features:
            if c not in X_pred.columns:
                X_pred[c] = 0
        X_pred = X_pred[features]
        proba = model.predict_proba(X_pred)[0][1]
        pred = model.predict(X_pred)[0]
        st.write(f"Predicted probability that case is closed: **{proba:.3f}**")
        st.write("Predicted class:", "Yes" if pred==1 else "No")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Model artifact saved at `/mnt/data/crime_model_artifacts.pkl` after training. You can download it from the environment if needed.")

if __name__ == '__main__':
    main()
