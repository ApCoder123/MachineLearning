import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree

from IPython.display import Image

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

print clf.predict([[5.1, 3.5, 1.4, 0.2]])

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
