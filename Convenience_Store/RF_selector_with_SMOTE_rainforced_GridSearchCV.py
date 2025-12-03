import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
#This is the new tool
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
import numpy as np
file_path = 'Data_for_Conven/FINAL_TRAINING_DATASET.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"error: replace {file_path} to your correct file path")
    exit()

print(f"sucessfully，total {len(df)} row.")

df['is_suitable_location'] = (df['store_count'] > 0).astype(int)

# Feature X muti - demension data
columns_to_drop = [
    'SA1_CODE_2021',
    'SA1_NAME_2021',
    'store_count',
    'is_suitable_location'
]
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=existing_columns_to_drop)
y = df['is_suitable_location']
# If some part is empty
X = X.fillna(0)

print(f"prepared feature X (there is {X.shape[1]} features) and y.")


#Shuffle
#80% train 20% for test
X_train, X_test, y_train, y_test = train_test_split(
    X,
         y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=42
)

print(f"Original training dataset shape: {pd.Series(y_train).value_counts()}")
print("\n start GridSearchCV")
print("WAIT...")

#SMOTE set up
smote_step = SMOTE(random_state=42)

#RF
model_step = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
#pipeline
pipeline = Pipeline(steps=[
    ('smote', smote_step),
    ('model', model_step)
])
#grid setting
param_grid = {
    'model__n_estimators': [10, 30, 50, 100, 200, 300, 400, 500],
    'model__max_depth': [10, 30, None],
    'model__min_samples_leaf': [1, 2]
}
#grid search method
grid_search = GridSearchCV(

    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("FINISHED！")

print("\nBest parameters found by GridSearchCV:")
print(grid_search.best_params_)
print(f"\nBest F1-score during Cross-Validation: {grid_search.best_score_:.4f}")
print("\nEvaluating the BEST model on the TEST SET:")

y_pred_best = grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

print("Confusion Matrix:")
print(f"       [Pred 0] [Pred 1]")
print(f"[True 0] {cm[0][0]:<8} {cm[0][1]:<8}")
print(f"[True 1] {cm[1][0]:<8} {cm[1][1]:<8}")
print("\n")

print(classification_report(
    y_test,
    y_pred_best,
    target_names=["Class 0 (Not suitable)", "Class 1 (Suitable)"]
))