import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
from sklearn import neighbors
from sklearn import grid_search
from sklearn import decomposition


#read from the csv file and return a Pandas DataFrame.
wine = pd.read_csv('wine.csv')

# The column names
original_headers = list(wine.columns.values)

# "quality" is the class attribute we are predicting.
class_column = 'quality'

#Choosing remaining attributes as feautures
feature_columns = original_headers[:-1] 

wine_feature = wine[feature_columns]
wine_feature = preprocessing.scale(wine_feature) #scaling the features
wine_class = wine[class_column]

#splitting the data for test and train purposes
train_feature, test_feature, train_class, test_class = \
    train_test_split(wine_feature, wine_class, stratify=wine_class, \
    train_size=0.75, test_size=0.25, random_state=None)

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)
knn.fit(train_feature, train_class)
print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))

#confusion matrix
prediction = knn.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
print("___________________________________________")

# Finding the best value for number of neighbours with 10-fold cross validation using Grid Search
k=np.arange(10)+1
parameters = {'n_neighbors':k}
knearest = sklearn.neighbors.KNeighborsClassifier()
clf = sklearn.grid_search.GridSearchCV(knearest,parameters, cv =10)
best_k=[]

# d-number of dimensions taken based on no. of parameters
for d in range(1,len(original_headers)-1): 
	svd = sklearn.decomposition.TruncatedSVD(n_components = d)
	feature_fit = svd.fit_transform(train_feature)
	clf.fit(feature_fit, train_class)
	best_k.append(clf.best_params_['n_neighbors']) #obtaining best k(neighbours) value for each value of no. of dimensions

n_neighbors = max(set(best_k), key=best_k.count) #choosing most repeated value for k 

#Applying 10-fold stratified cross-validation with best k-value(neighbours)
skf = StratifiedKFold(n_splits=10)
k_accuracies = []
kneighbour = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p=3)
for train, test in skf.split(wine_feature, wine_class):
	train_feature, test_feature = wine_feature[train], wine_feature[test]
	train_class, test_class = wine_class[train], wine_class[test]
	kneighbour.fit(train_feature, train_class)
	k_accuracies.append(knn.score(test_feature, test_class))

print("Test set accuracies with 10-fold cross-validation: ")
print([ '%.2f' % elem for elem in k_accuracies ]) 

print("Mean accuracy:")
print(round(np.mean(k_accuracies),2))


