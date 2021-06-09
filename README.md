# import tensorflow 
import tensorflow as tf

# import necessary files
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Decision tree Classifier
dt_clf = DecisionTreeClassifier(random_state=156)
iris_data = load_iris()
X_train , X_test, y_train ,y_test= train_test_split(iris_data.data, iris_data.target, test_size =0.2, random_state=11)

dt_clf.fit(X_train, y_train)

# import graphviz and assign tree.dot
from sklearn.tree import export_graphviz
export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names , \
feature_names = iris_data.feature_names, impurity=True, filled=True)

# build the tree using the decisiontree classifier

import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# import numpy and seaborn and graph function
import seaborn as sns
import numpy as np
%matplotlib inline

# feature importance
print("Feature importances:\n{0}".format(np.round(dt_clf. feature_importances_, 3)))

# create a bar graph using sns
for name, value in zip(iris_data.feature_names , dt_clf.feature_importances_):
  print('{0} : {1:.3f}'.format(name,value))

sns.barplot(x=dt_clf.feature_importances_ , y=iris_data.feature_names)
