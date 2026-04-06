import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

df = pd.read_csv("creditcard.csv")



scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df.drop(['Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    verbose=100,
    eval_metric='F1',
    class_weights=[1, 3]  
)

model.fit(X_train_res, y_train_res)


y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.9
y_pred = (y_prob > threshold).astype(int)


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
 
f1_fraud = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_prob)
print("\nF1-score for Fraud:", f1_fraud)
print("ROC-AUC:", auc)


joblib.dump({
    "model": model,
    "scaler": scaler
}, "model.pkl")
