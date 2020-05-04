from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB
bayes = MultinomialNB()
bayes.fit(X_train, y_train)

prediction_bayes = bayes.predict(X_test)
print(accuracy_score(prediction_bayes,y_test))
print(confusion_matrix(prediction_bayes, y_test, labels=[0, 1, 2]))