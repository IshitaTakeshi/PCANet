from os.path import exists
import pickle
import gzip
from urllib.request import urlretrieve

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from pcanet import PCANet


def load_mnist():
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    mnist_compressed = "mnist.pkl.gz"

    if not exists(mnist_compressed):
        print("Downloading MNIST")
        urlretrieve(url, mnist_compressed)

    # Load the dataset
    with gzip.open(mnist_compressed, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()

    data = [(X.reshape(-1, 28, 28), y) for X, y in data]
    return data


n_train = 1000
n_test = 1000

train_set, valid_set, test_set = load_mnist()

images_train, y_train = train_set
images_test, y_test = test_set

images_train, y_train = shuffle(images_train, y_train, random_state=0)
images_train, y_train = images_train[:n_train], y_train[:n_train]

images_test, y_test = shuffle(images_test, y_test, random_state=0)
images_test, y_test = images_test[:n_test], y_test[:n_test]


# digits = load_digits()
# images = digits.images
# y = digits.target

pcanet = PCANet(
    image_shape=28,
    filter_shape_l1=2, step_shape_l1=1, n_l1_output=4,
    filter_shape_l2=2, step_shape_l2=1, n_l2_output=4,
    block_shape=2
)
pcanet.validate_structure()

pcanet.fit(images_train)
X_train = pcanet.transform(images_train)
X_test = pcanet.transform(images_test)

model = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: " + str(accuracy))
