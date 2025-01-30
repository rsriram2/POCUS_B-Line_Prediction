import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

csv_file_path = "/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/NHANES/nhanes_data.csv"
data = pd.read_csv(csv_file_path)

data_above_50 = data[data['age'] > 50]

data_above_50 = data_above_50.dropna(subset=['event'])

X = data_above_50[['TMIMS', 'gender']]
Y = data_above_50['event']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#add age as a predictor

data_above_50 = data[data['age'] > 50].copy()


#map values
data_above_50.loc[:, 'gender_encoded'] = data_above_50['gender'].map({'Female': 0, 'Male': 1})

X = data_above_50[['TMIMS', 'gender_encoded']]
Y = data_above_50['event']

# Check for missing values and handle them
Y = Y.dropna()
X = X.loc[Y.index]

X = X.dropna()
Y = Y.loc[X.index]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


