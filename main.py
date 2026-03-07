import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

df = pd.read_csv("dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
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
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)