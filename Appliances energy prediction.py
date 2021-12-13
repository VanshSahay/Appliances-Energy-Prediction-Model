import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("energydata_complete.csv", sep=",")

data = data[["T1", "T2", "T3", "T4", "RH_1", "RH_2", "RH_3", "RH_4"]]

predict = "RH_4"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.01)

best = 0
'''for _ in range(10000):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    
    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train)
    acc = linear.score(X_test, Y_test)
    print(acc)

    if acc > best :
        best = acc
        with open("Appliances energy prediction.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("Appliances energy prediction.pickle", "rb")
linear = pickle.load(pickle_in)

print("Slope: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)


predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], Y_test[x])

p = 'T1'
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Predicted Humidity")
pyplot.show()

