from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pcanet import PCANet


digits = load_digits()
images = digits.images
y = digits.target

pcanet = PCANet((2, 2), (1, 1), 4,
                (2, 2), (1, 1), 4,
                (2, 2))

pcanet.fit(images)
X = pcanet.transform(images)
print("X.shape: " + str(X.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

model = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: " + str(accuracy))
