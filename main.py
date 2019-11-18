from myPackage import tools as tl
from myPackage import preprocess
from myPackage import minutiaeExtraction as minExtract
from enhancementFP import image_enhance as img_e
from os.path import basename, splitext, exists
import time
from numpy import mean, std
import os
from imutils import paths
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import _pickle as cPickle

import sklearn.ensemble
from sklearn import metrics
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2 as cv2
import os


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--path", required=True,
    #                 help="-p Source path where the images are stored.")
    # ap.add_argument("-r", "--results", required= False,
    #                 help="-r Destiny path where the results will be stored.")
    # args = vars(ap.parse_args())

    # Configuration

    EPOCHS = 100
    INIT_LR = 1e-3
    BS = 32
    image_ext = '.tif'
    plot = False
    path = None

    dir_gam = "./Image"
    dir_res = "E:\IBE\GitHub\MinutiaeFingerprint\Result"
    # ratio = 0.2
    # Create folders for results
    # -r ../Data/Results/fingerprints
    # if args.get("results") is not None:
    #     if not exists(args["results"]):
    #         tl.makeDir(args["results"])
    #     path = args["results"]
    path = dir_res
    dir = "E:\Aji Sapta Pramulen\Python\Kode Program\images\SI"
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(dir_gam)))
    # random.seed(42)
    # random.shuffle(imagePaths)
    # Extract names
    all_images = tl.natSort(imagePaths)
    data = []
    labels = []
    # Split train and test data
    # train_data, test_data = tl.split_train_test(all_images, ratio)
    print("\nAll_images size: {}\n".format(len(all_images)))
    all_times= []
    for image in all_images:
        start = time.time()
        name = splitext(basename(image))[0]
        print("\nProcessing image '{}'".format(name))
        cleaned_img = preprocess.blurrImage(image, name, plot)
        enhanced_img = img_e.image_enhance(cleaned_img, name, plot)
        cleaned_img = preprocess.cleanImage(enhanced_img, name, plot)
        # skeleton = preprocess.zhangSuen(cleaned_img, name, plot)
        skeleton = preprocess.thinImage(cleaned_img, name, plot)
        # minExtract.process(skeleton, name, plot, path)


        label = image.split(os.path.sep)[-2]
        temp = minExtract.process(skeleton, name, label)
        if label == "Accidental":
            label = 1
        elif (label == "Central pocket loop"):
            label = 2
        elif (label == "Leteral pocket loop"):
            label = 3
        elif (label == "Plain Arch"):
            label = 4
        elif (label == "Plain whorl"):
            label = 5
        elif (label == "Radial loop"):
            label = 6
        elif (label == "Tented arch"):
            label = 7
        elif (label == "Twinted loop"):
            label = 8
        elif (label == "Ulnair loop"):
            label = 9

        labels.append(label)

        # temp = minExtract.process(skeleton, name,label)
        data.append(temp)
        all_times.append((time.time()-start))
    print("[INFO] loading . ...")

    # data = np.array(data)
    # labels = np.array(labels)
    #
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    #
    # # Fitting Logistic Regression to the Training set
    # from sklearn.ensemble import RandomForestClassifier
    #
    # classifier = RandomForestClassifier(n_estimators=5000, criterion='entropy', random_state=0)
    # classifier.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_pred = classifier.predict(X_test)
    #
    # print('SkLearn : ', metrics.accuracy_score(y_test, y_pred))
    # print("Train Accuracy :: ", accuracy_score(y_train, classifier.predict(X_train)))


    # mean = mean(all_times)
    # std = std(all_times)
    # print("\n\nAlgorithm takes {:2.3f} (+/-{:2.3f}) seconds per image".format(mean, std))