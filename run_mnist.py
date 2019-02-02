import os
from os.path import isdir, join
import timeit
import argparse

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# avoid the odd behavior of pickle by importing under a different name
import pcanet as net
from utils import load_model, save_model, load_mnist, set_device


parser = argparse.ArgumentParser(description="PCANet example")
parser.add_argument("--gpu", "-g", type=int, default=-1,
                    help="GPU ID (negative value indicates CPU)")

subparsers = parser.add_subparsers(dest="mode",
                                   help='Choice of train/test mode')
subparsers.required = True
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--out", "-o", default="result",
                          help="Directory to output the result")

test_parser = subparsers.add_parser("test")
test_parser.add_argument("--pretrained-model", default="result",
                         dest="pretrained_model",
                         help="Directory containing the trained model")

args = parser.parse_args()


def train(train_set):
    images_train, y_train = train_set

    print("Training PCANet")

    pcanet = net.PCANet(
        image_shape=28,
        filter_shape_l1=2, step_shape_l1=1, n_l1_output=3,
        filter_shape_l2=2, step_shape_l2=1, n_l2_output=3,
        filter_shape_pooling=2, step_shape_pooling=2
    )

    pcanet.validate_structure()

    t1 = timeit.default_timer()
    pcanet.fit(images_train)
    t2 = timeit.default_timer()

    train_time = t2 - t1

    t1 = timeit.default_timer()
    X_train = pcanet.transform(images_train)
    t2 = timeit.default_timer()

    transform_time = t2 - t1

    print("Training the classifier")

    classifier = SVC(C=10)
    classifier.fit(X_train, y_train)
    return pcanet, classifier


def test(pcanet, classifier, test_set):
    images_test, y_test = test_set

    X_test = pcanet.transform(images_test)
    y_pred = classifier.predict(X_test)
    return y_pred, y_test


train_set, test_set = load_mnist()


if args.gpu >= 0:
    set_device(args.gpu)


if args.mode == "train":
    print("Training the model...")
    pcanet, classifier = train(train_set)

    if not isdir(args.out):
        os.makedirs(args.out)

    save_model(pcanet, join(args.out, "pcanet.pkl"))
    save_model(classifier, join(args.out, "classifier.pkl"))
    print("Model saved")

elif args.mode == "test":
    pcanet = load_model(join(args.pretrained_model, "pcanet.pkl"))
    classifier = load_model(join(args.pretrained_model, "classifier.pkl"))

    y_test, y_pred = test(pcanet, classifier, test_set)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: {}".format(accuracy))
