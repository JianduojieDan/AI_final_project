import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

print(f"training dataset size: {len(X_train)} | testing dataset size: {len(X_test)}")

print("\n start basic RF")
print("WAIT...")

#real RF
rf_baseline = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

rf_baseline.fit(X_train, y_train)

print("FINISHED！")

print("\n Here is the accuracy of the model:")

y_pred_baseline = rf_baseline.predict(X_test)

cm = confusion_matrix(y_test, y_pred_baseline)
print("Confusion Matrix:")
print(f"       [Pred 0] [Pred 1]")
print(f"[True 0] {cm[0][0]:<8} {cm[0][1]:<8}")
print(f"[True 1] {cm[1][0]:<8} {cm[1][1]:<8}")
print("\n")

print(classification_report(
    y_test,
    y_pred_baseline,
    target_names=["Class 0 (Not suitable)", "Class 1 (Suitable)"]
))