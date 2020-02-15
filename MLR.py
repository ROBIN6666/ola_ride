import pandas 
import numpy as np 
import matplotlib.pyplot as plt  
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('taxi.csv')
# print(data.head())

# split the data into the Dependent and Independent Variable.

x_data = data.iloc[:,0:-1].values # independent variables
y_data = data.iloc[:,-1].values # dependent variables

# print(y_data)

# spliting the data into train test split

X_train,X_test,Y_train,Y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=0)

#Applying the Model.

reg = LinearRegression()

reg.fit(X_train,Y_train)    

#To the performace of data Train and Test data sets

print("Train Score:", reg.score(X_train,Y_train))
print("Text Score:", reg.score(X_test,Y_test))

#using the Model.
pickle.dump(reg,open('taxi.pky','wb'))  

model = pickle.load(open('taxi.pky','rb'))

print(model.predict([[70, 2770000, 8000, 90]]))





