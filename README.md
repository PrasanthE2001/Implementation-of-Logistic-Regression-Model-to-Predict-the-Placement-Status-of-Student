# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries such as pandas module to read the corresponding csv file.
2.Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 
3.Import LabelEncoder and encode the corresponding dataset values. 
4.Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5.Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
6.Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by  : PRASANTH E
RegisterNumber: 22007885
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

![image](https://user-images.githubusercontent.com/114572171/201524873-634985db-7d4c-4f1b-ad91-7094ba11fd38.png)

![image](https://user-images.githubusercontent.com/114572171/201524879-b699782c-65c5-40cf-bab8-46aec74e95b2.png)

![image](https://user-images.githubusercontent.com/114572171/201524890-b5a1cc3d-9571-4827-a45f-afa732460870.png)

![image](https://user-images.githubusercontent.com/114572171/201524901-d3fd47f0-030a-465b-b3de-6d687da1ae4e.png)

![image](https://user-images.githubusercontent.com/114572171/201524914-0d477019-5f59-40fb-afb5-b8afc535fd5a.png)

![image](https://user-images.githubusercontent.com/114572171/201524947-e4110486-a47d-4bb2-8a06-edbdb78ed3e1.png)

![image](https://user-images.githubusercontent.com/114572171/201524963-66d39ebc-927c-47ca-b6a6-b5b9262eeec1.png)

![image](https://user-images.githubusercontent.com/114572171/201524971-eea0474b-9aa0-41f0-bc5b-3974571c2e0f.png)

![image](https://user-images.githubusercontent.com/114572171/201524978-8699e5e6-27fe-4c0c-a1ed-d1d51476f395.png)

![image](https://user-images.githubusercontent.com/114572171/201524990-dd1d49a5-f08f-4729-a794-f71dc2afab32.png)

![image](https://user-images.githubusercontent.com/114572171/201525178-085a972f-7076-49fd-b945-ad1535cf0d6e.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
