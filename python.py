#!/usr/bin/python



import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

a =int(input("Enter the index number of image: "))
data = pd.read_csv("/home/umesh/Downloads/train(1).csv").as_matrix()
clf = DecisionTreeClassifier()
xtrain = data[0:21000,1:]
train_label = data[0:21000,0]

clf.fit(xtrain,train_label)

xtest=data[21000:,1:]
actual_label = data[21000:,0]
# The input image idex.

d=xtest[a]

n =print(clf.predict( [d]))
d.shape =(28,28)
pt.imshow(d,cmap='gray')
print("Thank you very much!!, try one more...")
pt.show()
