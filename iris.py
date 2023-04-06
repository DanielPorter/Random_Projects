from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Loading and setting up variable names. 
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)

#Gives an accuracy score (knn score)
print("Accuracy:", accuracy)
plt.plot(accuracy)
plt.show()

#Change variable names so they are easier to understand
X = iris.data
y = iris.target

#Plotting and showing a graph
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset Scatter Plot')
plt.show()