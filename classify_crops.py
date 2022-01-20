#!/usr/bin/env python3
"""
Train and test supervised classification algorithm to learn Zindi's South Africa crop competition dataset
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from osgeo import gdal
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC


def preprocess_data(n_imgs=10**10,
                    train_size=10e-4):

    # shuffle the list of images
    train_data_dir = os.listdir(TRAIN_DATA_DIR)
    np.random.shuffle(train_data_dir)

    if len(train_data_dir) < n_imgs:
        n_imgs = len(train_data_dir)

    vh_arr_list = []
    vv_arr_list = []
    label_arr_list = []

    print("Extracting data arrays from imagery")
    for chip in train_data_dir[:n_imgs]:  # get only the first n images of training, test, and label data
        if chip.startswith('ref'):
            chip_idx = chip.find('s1_') + 3
            chip_no = chip[chip_idx: chip_idx + 4]
            for label_chip in os.listdir(TRAIN_LABEL_DIR):
                if label_chip.__contains__(chip_no):
                    vh_arr = gdal.Open(TRAIN_DATA_DIR + "/" + chip + "/VH.tif"
                                       ).ReadAsArray().astype("float32")
                    vv_arr = gdal.Open(TRAIN_DATA_DIR + "/" + chip + "/VV.tif"
                                       ).ReadAsArray().astype("float32")
                    label_arr = gdal.Open(TRAIN_LABEL_DIR + "/" + label_chip + "/labels.tif"
                                          ).ReadAsArray().astype("float32")
                    vh_arr_list.append(np.ndarray.flatten(vh_arr))
                    vv_arr_list.append(np.ndarray.flatten(vv_arr))
                    label_arr_list.append(np.ndarray.flatten(label_arr))
                    break

    assert len(vh_arr_list) == len(vv_arr_list) == len(label_arr_list), 'Train, test, and label data lengths vary'
    print("Training classification algorithm with", len(vh_arr_list), "random images from training dataset")

    vh_arrays, vv_arrays, label_arrays = [np.concatenate(arr)
                                          for arr in [vh_arr_list,
                                                      vv_arr_list,
                                                      label_arr_list]]
    X = np.array((vv_arrays, vh_arrays)).T  # array of independent variables with a vv column and a vh column
    y = np.array((label_arrays)).T  # array of labels corresponding to X

    print("Subdividing the training data into stratified training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)

    # X = np.array((np.ndarray.flatten(vv_arr), np.ndarray.flatten(vh_arr))).T
    # y = np.array((np.ndarray.flatten(label_arr))).T
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, stratify=y)

    print("Scaling the data")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, sc


def classify_random_forest(X_train,
                           y_train,
                           X_test,
                           y_test,
                           sc
                           ):
    # @params y_test, df, sc, band_list, and tif not used here but passed to other functions

    rf = RandomForestClassifier(min_weight_fraction_leaf=0.0,
                                criterion='gini',
                                max_leaf_nodes=None,
                                max_depth=None,
                                min_impurity_decrease=0,
                                bootstrap=True, oob_score=True,
                                n_jobs=-1,
                                random_state=29,
                                verbose=0,
                                warm_start=True)
    parameters = {'n_estimators': (150, 200, 250, 300, 350, 400, 450, 500),
                  'max_features': ('auto', 'log2', 'sqrt'),
                  'min_samples_split': (2, 5, 5, 10, 20, 30, 40, 50),
                  'min_samples_leaf': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
                  }
    sweep_rf = GridSearchCV(rf,
                            parameters,
                            scoring=None,
                            n_jobs=-1,
                            # iid=True,
                            refit=True,
                            cv=None,
                            verbose=1,
                            pre_dispatch='2*n_jobs',
                            error_score='raise',
                            return_train_score='warn')

    print('Fitting the random forest classification algorithm')
    sweep_rf.fit(X_train, y_train)

    print("Testing the trained random forest algorithm's class predictions")
    y_pred_rf = sweep_rf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_rf))

    cm = confusion_matrix(y_test, y_pred_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return sweep_rf


def classify_svm(X_train,
                 y_train,
                 X_test,
                 y_test,
                 sc):
    # @params y_test, sc, and tif not used here but passed to other functions

    # global y_pred
    # y_pred = sweep.predict(X_test)

    # train the SVM (not actually a sweep without GridSearchCV)
    sweep_svm = SVC(kernel='rbf',
                    C=5000,  # [5000,20000,50000]
                    gamma='auto')  # 'scale]'

    # sweep_svm = GridSearchCV(svm, parameters, scoring=None, n_jobs=-1, iid=True, refit=True,
    #                          cv=None, verbose=1, pre_dispatch='2*n_jobs', error_score='raise',
    #                          return_train_score='warn')

    print('Fitting the SVM')
    sweep_svm.fit(X_train, y_train)

    print("Testing the trained SVM algorithm's class predictions")
    y_pred_svm = sweep_svm.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_svm))

    cm = confusion_matrix(y_test, y_pred_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return sweep_svm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the SVM or Random Forest classification algorithm")
    parser.add_argument("-p",
                        "--path",
                        type=str,
                        default="/Users/kevin/zindi",
                        help="Enter the path to the directory containing Zindi's South "
                             "Africa crop competition train, test, and label data")
    parser.add_argument("-n",
                        "--n_imgs",
                        type=int,
                        default=10**3,
                        help="Number of satellite images to sample from")
    parser.add_argument("-f",
                        "--train_size",
                        type=float,
                        default=10e-4,
                        help="Fraction of training data to sample from")
    parser.add_argument("-rf",
                        "--random_forest",
                        type=bool,
                        default=False,
                        help="Train the supervised classification algorithm using the random forest classification")
    parser.add_argument("-svm",
                        type=bool,
                        default=False,
                        help="Train the supervised classification algorithm using a support vector machine")
    args = parser.parse_args()
    os.chdir(args.path)
    # os.chdir("/Users/kevin/zindi")

    TRAIN_DATA_DIR = "ref_south_africa_crops_competition_v1_train_source_s1"
    TRAIN_LABEL_DIR = "ref_south_africa_crops_competition_v1_train_labels"
    TEST_DATA_DIR = "ref_south_africa_crops_competition_v1_test_source_s1"
    TEST_LABEL_DIR = "ref_south_africa_crops_competition_v1_test_labels"

    X_train, X_test, y_train, y_test, sc = preprocess_data(args.n_imgs, args.train_size)
    if args.svm:
        svm_sweep = classify_svm(X_train, y_train, X_test, y_test, sc)
    else:
        rf_sweep = classify_random_forest(X_train, y_train, X_test, y_test, sc)
