import os
from os.path import isdir, join
import timeit
import argparse

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from pcanet import PCANet
from utils import load_model, save_model, load_mnist


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
                         help="Directory to the trained model")

args = parser.parse_args()


def train(train_set):
    images_train, y_train = train_set

    pcanet = PCANet(
        image_shape=28,
        filter_shape_l1=5, step_shape_l1=1, n_l1_output=8,
        filter_shape_l2=5, step_shape_l2=1, n_l2_output=4,
        filter_shape_pooling=5, step_shape_pooling=5
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
    from cupy.cuda import Device
    Device(args.gpu).use()
    print("GPU {} is in use".format(args.gpu))


if args.mode == "train":
    print("Training the model...")
    pcanet, classifier = train(train_set)

    if not isdir(args.out):
        os.makedirs(args.out)

    save_model(pcanet, join(args.out, "pcanet.pkl"))
    save_model(classifier, join(args.out, "classifier.pkl"))

elif args.mode == "test":
    pcanet = load_model(join(args.pretrained_model, "pcanet.pkl"))
    classifier = load_model(join(args.pretrained_model, "classifier.pkl"))

    y_test, y_pred = test(pcanet, classifier, test_set)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: {}".format(accuracy))
