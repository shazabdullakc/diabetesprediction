import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
df=pd.read_csv("Part2\diabetes.csv")
features=df[df.columns[:-1]].values
labels=df[df.columns[-1]].values
xtrain,xtemp,ytrain,ytemp=train_test_split(features,labels,test_size=.3,random_state=0)
xvalid,xtest,yvalid,ytest=train_test_split(xtemp,ytemp,test_size=.5,random_state=0)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xvalid=sc.transform(xvalid)
xtest=sc.transform(xtest)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation="relu"),
    tf.keras.layers.Dense(16,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.fit(xtrain, ytrain, batch_size=16, epochs=20, validation_data=(xvalid, yvalid))
print("Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome")
userinput=[]
for feature in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]:
    value=int(input(f"enter the {feature} value:"))
    userinput.append(value)
userinputarray=np.array(userinput).reshape(1,-1)
scaled_input=sc.transform(userinputarray)

result=model.predict(scaled_input)
print(f"diabetic,accuracy:{result}" if result>=0.5 else print(f"not diabetic,accuracy:{1-result}"))