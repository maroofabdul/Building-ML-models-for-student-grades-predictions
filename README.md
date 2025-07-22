# Building-ML-models-for-student-grades-predictions
Build ML two models SVM and Logistic Regression for Student Grades predictions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def categorize(score):
    if score < 10:
        return 'Low'
    elif 10 <= score < 15:
        return 'Medium'
    else:
        return 'High'

df_cleaned['performance'] = df_cleaned['G3'].apply(categorize)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_cleaned['performance'] = le.fit_transform(df_cleaned['performance'])
columns_to_drop = ['address',"school","paid","higher", 'reason', 'guardian', 'romantic', 'famsup', 'nursery', 'goout', 'Walc', 'famrel']
df_cleaned = df_cleaned.drop(columns=columns_to_drop)
print("Remaining columns after dropping:")
print(df_cleaned.columns.tolist())
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['sex', 'famsize',"Mjob","Fjob","schoolsup", 'Pstatus', 'activities', 'internet']
le = LabelEncoder()
for col in categorical_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
print(df_cleaned[categorical_cols].head())
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['sex', 'famsize',"Mjob","Fjob","schoolsup", 'Pstatus', 'activities', 'internet']
le = LabelEncoder()
for col in categorical_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
print(df_cleaned[categorical_cols].head())
import seaborn as sns
import matplotlib.pyplot as plt
selected_features = ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']
sns.pairplot(df_cleaned[selected_features], corner=True)
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()
categorical_plot_features = ['studytime', 'failures', 'Dalc',"traveltime"]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axs = axs.flatten()
for i, feature in enumerate(categorical_plot_features):
    sns.barplot(x=feature, y='G3', data=df_cleaned, ax=axs[i], palette='viridis')
    axs[i].set_title(f'Average G3 vs {feature}')
    axs[i].set_ylabel('Average Final Grade (G3)')
if len(categorical_plot_features) % 2 != 0:
    fig.delaxes(axs[-1])

plt.tight_layout()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
numeric_df = df_cleaned.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
df_cleaned.to_csv("student_data_preprocessed.csv", index=False)
from google.colab import files
files.download("student_data_preprocessed.csv")
df_cleaned.head()
df_cleaned.columns
from sklearn.model_selection import train_test_split

X = df_cleaned.drop('performance', axis=1)
y = df_cleaned['performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("🔹 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_log, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_log, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("🔹 SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_svm, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_svm, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

print(X_test.head())
print("Logistic Regression Prediction:", log_reg.predict(X_test.head()))
print("SVM Prediction:", svm.predict(X_test.head()))

import pandas as pd
import numpy as np
test_samples = pd.DataFrame({
    'sex': [1, 0, 1, 0, 1],
    'age': [18, 20, 17, 15, 16],
    'famsize': [1, 0, 1, 1, 0],
    'Pstatus': [1, 0, 1, 1, 0],
    'Medu': [4, 2, 1, 3, 0],
    'Fedu': [4, 2, 2, 1, 0],
    'Mjob': [2, 3, 1, 0, 4],
    'Fjob': [3, 2, 1, 4, 0],
    'traveltime': [1, 2, 1, 1, 3],
    'studytime': [2, 3, 1, 2, 4],
    'failures': [0, 1, 2, 0, 3],
    'schoolsup': [0, 1, 1, 0, 1],
    'activities': [1, 0, 1, 0, 1],
    'internet': [1, 0, 1, 1, 0],
    'freetime': [3, 2, 4, 3, 1],
    'Dalc': [1, 2, 1, 3, 4],
    'health': [4, 3, 5, 2, 1],
    'absences': [2, 4, 6, 10, 0],
    'G1': [14, 10, 8, 15, 5],
    'G2': [15, 11, 7, 14, 6],
    'G3': [16, 12, 6, 13, 5]
})
print("=== Logistic Regression ===")
print(log_reg.predict(test_samples))
print("\n=== Support Vector Machine (SVM) ===")
print(svm.predict(test_samples))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(name, model, X_test, y_true):
    y_pred = model.predict(X_test)

    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


X_custom = test_samples
y_true = [2, 1, 0, 2, 0]
evaluate_model("Logistic Regression", log_reg, X_custom, y_true)
evaluate_model("SVM", svm, X_custom, y_true)

