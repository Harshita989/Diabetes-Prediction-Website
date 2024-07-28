from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')
def result(request):
    data=pd.read_csv('diabetes_dataset.csv')
    X=data.drop('Outcome',axis=1)
    Y=data['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    val1=float(request.GET['n1'])
    val2=float(request.GET['n2'])
    val3=float(request.GET['n3'])
    val4=float(request.GET['n4'])
    val5=float(request.GET['n5'])
    val6=float(request.GET['n6'])
    val7=float(request.GET['n7'])
    val8=float(request.GET['n8'])
    pred=rf_model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    result2=""
    if pred==[1]:
        result2="Positive"
    else:
        result2="Negative"
    
    return render(request,'predict.html',{"result2":result2})