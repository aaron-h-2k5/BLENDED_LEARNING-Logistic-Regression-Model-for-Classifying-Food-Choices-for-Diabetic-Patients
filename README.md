# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients
### Developed by: Aaron H
### RegisterNumber:  2122230400001
## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Dataset**  
   Import and load the dataset for analysis.

2. **Data Preprocessing**  
   Clean and preprocess the data to ensure consistency and accuracy.

3. **Split Data into Training and Testing Sets**  
   Divide the dataset into training and testing subsets for model validation.

4. **Train Logistic Regression Model**  
   Use the training data to fit and train a logistic regression model.

5. **Generate Predictions**  
   Apply the trained model to predict outcomes on the testing data.

6. **Evaluate Model Performance**  
   Assess the model’s accuracy and performance metrics.

7. **Visualize Results**  
   Create visualizations to interpret and present the model's outcomes.

8. **Make Predictions on New Data**  
   Utilize the trained model to make predictions on fresh or unseen data.

## Program:
```
# Program to implement Logistic Regression for classifying food choices based on nutritional information.

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
df = pd.read_csv(url)

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Encode the target column ('class') into binary labels
# Assuming 'Diabetic' means classifying 'In Moderation' as 1 (diabetic-friendly) and others as 0
df['class'] = df['class'].str.strip("'")  # Remove surrounding quotes
df['Diabetic'] = df['class'].apply(lambda x: 1 if x == 'In Moderation' else 0)

# Define features and target
X = df.drop(columns=['class', 'Diabetic'])  # Features
y = df['Diabetic']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Output:
<img width="697" alt="Screenshot 2024-11-17 at 8 05 07 PM" src="https://github.com/user-attachments/assets/abde9e60-445e-46ab-98c0-a1a16969ed01">


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
