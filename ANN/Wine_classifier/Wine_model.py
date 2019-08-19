# coding=utf-8
"""Sklearn wine dataset classifier."""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from DenseClassifier import ANNClassifier
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier


def eval_metric(confusion_mx):
    print(confusion_mx)
    base = confusion_mx[0][1] + sum(confusion_mx[1])
    if base:
        print("{0:.2f}%".format((confusion_mx[1][1] / base) * 100))
    else:
        print("100.00%")


if __name__ == "__main__":
    features, target = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30)
    model = ExtraTreesClassifier(n_estimators=50, max_depth=None,
                         min_samples_split=2, criterion="gini")
    model = make_pipeline(StandardScaler(), PCA(n_components=6),
                          model)
    model.fit(X_train, y_train)
    print("Test")
    eval_metric(confusion_matrix(y_test, model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, model.predict(X_train).round()))
    input()
    print("LGBM model training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    param = {'objective': 'binary', "learning_rate": 0.01}
    param['metric'] = ['accuracy']
    num_round = 500
    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[lgb.Dataset(X_test, y_test)])
    print("Test")
    eval_metric(confusion_matrix(y_test, lgb_model.predict(X_test).round()))
    print("Training")
    eval_metric(confusion_matrix(y_train, lgb_model.predict(X_train).round()))

    y_train = to_categorical(y_train)
    model = make_pipeline(StandardScaler(), PCA(n_components=6),
                          ANNClassifier(6, 3, architecture=[(7, 4)], regularize=True, EPOCHS=1000))
    model.fit(X_train, y_train)
    print(confusion_matrix(y_test, model.predict(X_test).argmax(axis=1)))
