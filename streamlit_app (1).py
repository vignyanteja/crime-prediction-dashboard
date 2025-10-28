"""
streamlit_app.py
Patched Streamlit dashboard for Crime Prediction (polished multi-section layout).
- Uses relative dataset path: 'crime_dataset_india.csv'
- Features (as provided): 'Victim Age', 'Victim Gender', 'Weapon Used', 'Police Deployed', 'Occurrence Hou'
- Target: 'Case Closed'
- Safe preprocessing and robust OneHotEncoder usage (supports different sklearn versions)
- Interactive Plotly charts and Folium map (streamlit-folium)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_FILE = "crime_dataset_india.csv"

st.set_page_config(page_title="Crime Prediction & Analysis", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    # Trim column names
    df.columns = [c.strip() for c in df.columns]
    # Parse dates where present
    for col in ["Date Reported", "Date of Occurrence", "Date Case Closed"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    # Time column to hour if exists
    if "Time of Occurrence" in df.columns:
        df["Occ_Hour"] = pd.to_datetime(df["Time of Occurrence"], format="%H:%M", errors="coerce").dt.hour
    elif "Occurrence Hou" in df.columns:
        df["Occ_Hour"] = pd.to_numeric(df["Occurrence Hou"], errors="coerce")
    elif "Occ_Hour" not in df.columns:
        df["Occ_Hour"] = np.nan
    # Clean victim age
    if "Victim Age" in df.columns:
        df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    # Target binary
    if "Case Closed" in df.columns:
        df["Case_Closed_Flag"] = df["Case Closed"].astype(str).str.strip().str.lower().apply(lambda x: 1 if x=="yes" else 0)
    else:
        df["Case_Closed_Flag"] = 0
    return df

def make_encoder(cat_cols):
    # Create OneHotEncoder compatible with sklearn version differences
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return enc

@st.cache_data(show_spinner=False)
def train_model(df, random_state=42):
    df = df.copy()
    target = "Case_Closed_Flag"
    # Define features (matching user's provided exact names)
    feature_names = ["Victim Age", "Victim Gender", "Weapon Used", "Police Deployed", "Occurrence Hou", "Occ_Hour"]
    # Keep only available features
    features = [f for f in feature_names if f in df.columns]
    if len(features) == 0:
        raise ValueError("No matching features found in the dataset. Expected columns like: " + ", ".join(feature_names))
    # Prepare X and y
    X = df[features].copy()
    y = df[target].fillna(0).astype(int)
    # Standardize columns: categorical -> str, numeric -> numeric
    cat_cols = []
    num_cols = []
    for col in X.columns:
        if X[col].dtype == 'O' or col in ["Victim Gender", "Weapon Used"]:
            X[col] = X[col].fillna("Unknown").astype(str)
            cat_cols.append(col)
        else:
            # numeric coercion for numeric-like columns
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            num_cols.append(col)
    # Fit encoder on categorical cols
    encoder = None
    X_cat_enc = pd.DataFrame(index=X.index)
    if len(cat_cols) > 0:
        encoder = make_encoder(cat_cols)
        X_cat_enc = pd.DataFrame(encoder.fit_transform(X[cat_cols]), index=X.index, columns=encoder.get_feature_names_out(cat_cols))
    # Combine numeric and encoded categorical
    X_num = X[num_cols].reset_index(drop=True) if len(num_cols)>0 else pd.DataFrame(index=X.index)
    X_final = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)
    # Train/test split
    if len(X_final) < 10:
        raise ValueError("Not enough data to train model after preprocessing.")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y if len(np.unique(y))>1 else None, random_state=random_state)
    # Model
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    artifacts = {
        "model": model,
        "encoder": encoder,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_columns": X_final.columns.tolist()
    }
    return artifacts, acc, report, cm

# Small city coords for map visualization
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
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["Overview", "Exploration", "Model", "Predict", "Map"])

    st.title("Crime Prediction & Analysis — Polished Dashboard")
    st.markdown("This app uses a fixed dataset and trains a RandomForest model to predict whether a case is closed (`Case Closed: Yes/No`).")

    # Load and preprocess
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset file '{DATA_FILE}' not found. Please upload it to the app folder.")
        return

    df = preprocess(df)

    # Sidebar filters
    st.sidebar.markdown("### Filters")
    cities = sorted(df['City'].dropna().unique().tolist()) if 'City' in df.columns else []
    city_filter = st.sidebar.multiselect("Cities", options=cities, default=[])
    if city_filter:
        df = df[df['City'].isin(city_filter)]

    if section == "Overview":
        st.header("Dataset Overview")
        st.dataframe(df.head(10))
        st.write("Summary statistics:")
        st.write(df.describe(include='all').transpose())

    elif section == "Exploration":
        st.header("Exploratory Visualizations")
        if 'Crime Description' in df.columns:
            top_n = st.slider("Top N crime types", 5, 30, 10)
            counts = df['Crime Description'].value_counts().nlargest(top_n).reset_index()
            counts.columns = ['Crime Description','Count']
            fig = px.bar(counts, x='Crime Description', y='Count', title=f"Top {top_n} Crime Descriptions")
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'Crime Description' column found for this view.")

        if 'Victim Gender' in df.columns:
            gender_counts = df['Victim Gender'].fillna("Unknown").value_counts().reset_index()
            gender_counts.columns = ['Gender','Count']
            fig2 = px.pie(gender_counts, names='Gender', values='Count', title="Victim Gender Distribution")
            st.plotly_chart(fig2, use_container_width=True)

    elif section == "Model":
        st.header("Machine Learning — Predict Case Closed (Yes/No)")
        st.markdown("Training a RandomForest on structured features. Click 'Train model' to (re)train.")

        if st.button("Train model"):
            with st.spinner("Training model..."):
                try:
                    artifacts, acc, report, cm = train_model(df)
                    joblib.dump(artifacts, "crime_model_artifacts.pkl")
                    st.success(f"Model trained. Accuracy: {acc:.3f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        else:
            # Try to train once (cached) for display, but catch errors
            try:
                artifacts, acc, report, cm = train_model(df)
            except Exception as e:
                st.error(f"Training failed: {e}")
                artifacts = None
                acc = None
                report = None
                cm = None

        if acc is not None:
            st.subheader("Performance")
            st.metric("Accuracy", f"{acc:.3f}")
            if cm is not None:
                cm_df = pd.DataFrame(cm, index=["Actual_No","Actual_Yes"], columns=["Pred_No","Pred_Yes"])
                st.write("Confusion matrix:")
                st.dataframe(cm_df)
            if report is not None:
                st.write("Classification report:")
                st.dataframe(pd.DataFrame(report).transpose().round(3))

    elif section == "Predict":
        st.header("Make a prediction (single case)")
        # Load artifacts if available or train on the fly
        try:
            artifacts = joblib.load("crime_model_artifacts.pkl")
        except Exception:
            try:
                artifacts, _, _, _ = train_model(df)
            except Exception as e:
                st.error(f"Model not available: {e}")
                artifacts = None

        if artifacts is not None:
            model = artifacts["model"]
            encoder = artifacts.get("encoder", None)
            cat_cols = artifacts.get("cat_cols", [])
            num_cols = artifacts.get("num_cols", [])
            feature_columns = artifacts.get("feature_columns", [])

            with st.form("predict_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    age = st.number_input("Victim Age", min_value=0, max_value=120, value=30)
                    gender = st.selectbox("Victim Gender", options=sorted(df['Victim Gender'].fillna("Unknown").unique().tolist()) if 'Victim Gender' in df.columns else ["Unknown"])
                with col2:
                    weapon = st.selectbox("Weapon Used", options=sorted(df['Weapon Used'].fillna("Unknown").unique().tolist()) if 'Weapon Used' in df.columns else ["Unknown"])
                    police_deployed = st.number_input("Police Deployed", min_value=0, max_value=1000, value=5)
                with col3:
                    occ_hour = st.slider("Occurrence hour (0-23)", 0, 23, 12)
                    submit = st.form_submit_button("Predict")
            if submit:
                # Build input df according to expected cols (using exact names)
                input_df = pd.DataFrame([{
                    "Victim Age": age,
                    "Victim Gender": gender,
                    "Weapon Used": weapon,
                    "Police Deployed": police_deployed,
                    "Occurrence Hou": occ_hour,
                    "Occ_Hour": occ_hour
                }])
                # Preprocess same as training
                # Separate cat and num
                for col in input_df.columns:
                    if col in cat_cols:
                        input_df[col] = input_df[col].astype(str).fillna("Unknown")
                    else:
                        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
                # Encode
                X_num = input_df[num_cols].reset_index(drop=True) if len(num_cols)>0 else pd.DataFrame(index=input_df.index)
                if encoder is not None and len(cat_cols)>0:
                    x_cat_enc = pd.DataFrame(encoder.transform(input_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
                else:
                    x_cat_enc = pd.DataFrame(index=input_df.index)
                X_pred = pd.concat([X_num.reset_index(drop=True), x_cat_enc.reset_index(drop=True)], axis=1)
                # Align columns used in training
                for c in feature_columns:
                    if c not in X_pred.columns:
                        X_pred[c] = 0
                X_pred = X_pred[feature_columns]
                proba = model.predict_proba(X_pred)[0][1] if hasattr(model, "predict_proba") else None
                pred = model.predict(X_pred)[0]
                st.write(f"Predicted probability that case is closed: **{proba:.3f}**" if proba is not None else "Probability not available")
                st.write("Predicted class:", "Yes" if int(pred)==1 else "No")
        else:
            st.info("Model not trained yet. Train the model in the 'Model' section.")

    elif section == "Map":
        st.header("Map — Top Cities")
        if 'City' in df.columns:
            city_counts = df['City'].value_counts().nlargest(20).reset_index()
            city_counts.columns = ['City','Count']
            city_counts = city_counts[city_counts['City'].isin(CITY_COORDS.keys())]
            if not city_counts.empty:
                m = folium.Map(location=[20.5937,78.9629], zoom_start=5)
                for _, row in city_counts.iterrows():
                    c = row['City']
                    cnt = int(row['Count'])
                    lat, lon = CITY_COORDS[c]
                    folium.CircleMarker(location=[lat, lon],
                                        radius=4 + np.log1p(cnt),
                                        popup=f"{c}: {cnt} incidents").add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.info("No cities with known coordinates in top 20. Extend CITY_COORDS to include more cities.")
        else:
            st.info("No 'City' column found for map visualization.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Auto-generated dashboard — edit streamlit_app.py to customize.")

if __name__ == '__main__':
    main()
