import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading split datasets...")
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

# Separate features (X) and target labels (y)
X_train, y_train_raw = train.drop('label', axis=1), train['label']
X_val, y_val_raw = val.drop('label', axis=1), val['label']
X_test, y_test_raw = test.drop('label', axis=1), test['label']

print("Encoding text labels to numbers...")
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_val = le.transform(y_val_raw)
y_test = le.transform(y_test_raw)
num_classes = len(le.classes_)

# IMPORTANT: Save the LabelEncoder so your prediction script knows what "0" or "1" means later!
joblib.dump(le, 'label_encoder.pkl')
print("Saved label_encoder.pkl")

print("Setting up LightGBM datasets...")
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'verbose': -1
}

print("Training model...")
bst = lgb.train(
    params, 
    train_data, 
    num_boost_round=200, 
    valid_sets=[train_data, valid_data], 
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Save the trained AI model
model_filename = 'motor_fault_model.txt'
bst.save_model(model_filename)
print(f"\nModel successfully saved as: {model_filename}")

print("\nEvaluating model on UNSEEN TEST DATA...")
y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n--- Final Test Set Performance ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))