import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the datasets
train_df = pd.read_csv('/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/lr_train_data.csv')
test_df = pd.read_csv('/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/lr_test_data.csv')
val_df = pd.read_csv('/dcs05/ciprian/smart/pocus/rushil/POCUS_B-Line_Prediction/create_image_metrics/lr_val_data.csv')

# Select the mean metrics as features
features = ['mean_mean', 'mean_max', 'mean_min']

# Extract features and target variable from the datasets
X_train = train_df[features].values
y_train = train_df['Label_aggregate_label'].apply(lambda x: 1 if x == 'b-line' else 0).values
X_test = test_df[features].values
y_test = test_df['Label_aggregate_label'].apply(lambda x: 1 if x == 'b-line' else 0).values
X_val = val_df[features].values
y_val = val_df['Label_aggregate_label'].apply(lambda x: 1 if x == 'b-line' else 0).values

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")
