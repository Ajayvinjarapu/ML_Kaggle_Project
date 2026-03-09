import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

st.title("Heart Disease Prediction ML App")

df = pd.read_csv("dataset.csv")

st.sidebar.header("Model Controls")

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
run_model = st.sidebar.button("Run Model")

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop("target", axis=1)
y = df["target"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

if run_model:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators":[50,100],
        "model__max_depth":[3,5,7]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, y_train)

    pred = grid.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    st.subheader("Model Results")

    st.write("Best Parameters:", grid.best_params_)
    st.write("Accuracy:", accuracy)
    st.write("Mean Squared Error:", mse)
    st.write("R2 Score:", r2)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, pred))

    st.subheader("Heart Disease Prediction Result")

    sample_input = pd.DataFrame([X_test.iloc[0]], columns=X.columns)
    prediction = grid.predict(sample_input)

    if prediction[0] == 1:
        st.error("The model predicts that the individual may have Heart Disease.")
    else:
        st.success("The model predicts that the individual does NOT have Heart Disease.")