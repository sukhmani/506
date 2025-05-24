from sklearn import  datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
#-----------------------------------------------------------------------------------------------------------------------
# Problem 2: The data contains 4 different features namely sepal length, sepal width, petal length, and petal width
# It is important to recognize which feature set(s) performs the best. Choose the best combination based on your
# experiment. You will have 10 different combinations possible (e.g. {SL, SW, PL, PW, (SL,SW), (SL, PL)...(SL, SW, PL, PW)} 
# You will show different performance after based on the best training parameters from Problem #1.
# This may show that the more number of features doesn't end up with better accuracy necessarily. 
# You will have the performance results per 10 combinations and plot the results on a graph for each classifier.
# with your analysis in words.
# Refer to the topic in the "curse of dimensionality"
# Resource : https://en.wikipedia.org/wiki/Curse_of_dimensionality
#-----------------------------------------------------------------------------------------------------------------------

x=iris.data   # data that contains 4 features of 150 samples. 
y=iris.target # labels with ground truth information

# split the data into split% training and (100-split)% testing
split = 0.9

#-----------------------------------------------------------------------------------------
# Problem 3: Once you decide the best feature set(s) from the Problem #2, it is important to recognize 
# how the size of training set versus testing set (or ratio between sets) would influence the
# overall representative performance. You will have the performance results per 10%, 20% ... 90% and plot
# the results on a graph for each classifier with your analysis in words.
#-----------------------------------------------------------------------------------------
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=split)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#-----------------------------------------------------------------------------------------
# Problem 1: Write a program such that different training variables
# such as "mxdepth" for DecisionTreeClassifier and "k" for KNN classifier 
# can have consecutive values being experimented. For example, rewrite the
# following code so that mxdepth starts from 1 to 10 or k goes from 1 to 10
# You will need to plot the accuracy per varying these parameters of each classifier 
# with your analysis in words
#-----------------------------------------------------------------------------------------
# Resource:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# A decision tree classifier. Training parameter is mxdepth
mxdepth = 1
classifier1 = DecisionTreeClassifier(max_depth=mxdepth)
classifier1.fit(x_train,y_train)
predictions=classifier1.predict(x_test)
print(f"DTC ({mxdepth}) = %0.2f accuracy" % accuracy_score(y_test,predictions))

# Resource:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# Finds the K-neighbors of a point. Returns indices of and distances to the neighbors of each point.
# Training parameter is k
k = 1
classifier2 = KNeighborsClassifier(n_neighbors=k)
classifier2.fit(x_train,y_train)
predictions=classifier2.predict(x_test)
print(f"KNN ({k}) = %0.2f accuracy" % accuracy_score(y_test,predictions))

