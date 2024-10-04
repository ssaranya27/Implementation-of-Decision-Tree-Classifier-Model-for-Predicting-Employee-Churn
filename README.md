# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection and Preprocessing: Collect and preprocess data (handle missing values, encode categories, scale features).

2.Data Splitting: Split data into training and testing sets.

3.Model Building: Train a Decision Tree Classifier on the training data.

4.Model Evaluation: Test the model on the testing set and evaluate its performance (accuracy, precision, recall).

## Program:

```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SARANYA S.
RegisterNumber:  212223220101


import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()##check null values
df["left"].value_counts()#count values
df["last_evaluation"].value_counts() #count values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['salary']=le.fit_transform(df['salary'])
df.head()
x = df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df[["left"]]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
dt.predict([[1,2,3,4,5,6,7,8]])

```



## Output:
X value:

![image](https://github.com/user-attachments/assets/42bab4c2-8dd2-4fca-98c4-102cbcdf485b)

Y value:

![image](https://github.com/user-attachments/assets/bdec9c2c-2d30-485a-94a7-42e3a957475a)

Accuracy:

![image](https://github.com/user-attachments/assets/647f1a7d-3f2c-499e-a797-cca2a039e7ba)

Predicted value:

![image](https://github.com/user-attachments/assets/782905b0-6429-4aee-9a4d-ad304ff20cf1)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
