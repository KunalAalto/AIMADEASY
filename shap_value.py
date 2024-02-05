# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

# COMMAND ----------

folder_path = # insert your path here.

# COMMAND ----------

data= pd.read_csv(folder_path + 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = data.set_index('customerID')

# COMMAND ----------

data.isnull().any().sum()
# Data has no null values.

# COMMAND ----------

df = data.copy()

# COMMAND ----------

df.head()

# COMMAND ----------

df['TotalCharges'] = df['TotalCharges'].replace(' ', 0.0)


# COMMAND ----------

df['TotalCharges'] = df['TotalCharges'].astype('float64')


# COMMAND ----------

df['TotalCharges'].isnull().any()


# COMMAND ----------



# COMMAND ----------

categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols


# COMMAND ----------

numeric_col = data.select_dtypes(include=['int', 'float64']).columns
numeric_col

# COMMAND ----------

data.describe()

# COMMAND ----------

binary_cat = []
other_cat =[]
for col in categorical_cols:
    categories = df[col].unique()
    #print(f"Unique categories in '{col}': {categories}")
    if len(categories)==2:
        binary_cat.append(col)
    else:
        other_cat.append(col)
print("Binary categorical columns:", binary_cat)
print("Other categorical columns >2:", other_cat)


# COMMAND ----------

# replacing the binary category in each columns
df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})
df['Partner'] = df['Partner'].replace({'Yes':1, 'No':0})
df['Dependents'] = df['Dependents'].replace({'Yes':1, 'No':0})
df['PhoneService'] = df['PhoneService'].replace({'Yes':1, 'No':0})
df['PaperlessBilling'] = df['PaperlessBilling'].replace({'Yes':1, 'No':0})
df['Churn'] = df['Churn'].replace({'Yes':1, 'No':0})

# COMMAND ----------

# One hot encoding
df_encoded = pd.get_dummies(df, columns=other_cat)


# COMMAND ----------

input_features =['gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'tenure',
 'PhoneService',
 'PaperlessBilling',
 'MonthlyCharges',
 'TotalCharges',
 'MultipleLines_No',
 'MultipleLines_No phone service',
 'MultipleLines_Yes',
 'InternetService_DSL',
 'InternetService_Fiber optic',
 'InternetService_No',
 'OnlineSecurity_No',
 'OnlineSecurity_No internet service',
 'OnlineSecurity_Yes',
 'OnlineBackup_No',
 'OnlineBackup_No internet service',
 'OnlineBackup_Yes',
 'DeviceProtection_No',
 'DeviceProtection_No internet service',
 'DeviceProtection_Yes',
 'TechSupport_No',
 'TechSupport_No internet service',
 'TechSupport_Yes',
 'StreamingTV_No',
 'StreamingTV_No internet service',
 'StreamingTV_Yes',
 'StreamingMovies_No',
 'StreamingMovies_No internet service',
 'StreamingMovies_Yes',
 'Contract_Month-to-month',
 'Contract_One year',
 'Contract_Two year',
 'PaymentMethod_Bank transfer (automatic)',
 'PaymentMethod_Credit card (automatic)',
 'PaymentMethod_Electronic check',
 'PaymentMethod_Mailed check']

# COMMAND ----------

target = 'Churn'

# COMMAND ----------

X = df_encoded[input_features]
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# COMMAND ----------

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')




# COMMAND ----------

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)

# COMMAND ----------

feature_importances = rf_classifier.feature_importances_
feature_importances

# COMMAND ----------

explainer = shap.TreeExplainer(rf_classifier)


# COMMAND ----------

shap_values = explainer.shap_values(X_test)


# COMMAND ----------

shap.summary_plot(shap_values[1], X_test)


# COMMAND ----------


