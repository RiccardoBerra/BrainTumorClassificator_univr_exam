import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
import cv2
import glob



def knn(trainData, valData, trainLabels, valLabels, testLabels) -> None:
    kVals = range(1, 10, 2)
    accuracies = []

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 10, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)

        # evaluate the model and update the accuracies list
        score = model.score(valData, valLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))

    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    # show a final classification report demonstrating the accuracy of the classifier

    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))

    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(50,)))):
        # grab the image and classify it
        image = testData[i]
        prediction = model.predict(image.reshape(1, -1))[0]

        image = image.reshape((240, 240)).astype("uint8")

        # show the prediction
        print("Has Brain tumor: {}  Test Labels : {}".format(prediction, testLabels[i]))
        cv2.imshow("Image ", image)
        cv2.waitKey(0)


def svm(x_train, x_test, y_train, y_test) -> None:
    from sklearn import svm

    # Creating a support vector classifier
    max_iteration = 1000

    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        print('----', kernel, '----')
        model = svm.SVC(kernel=kernel, max_iter=max_iteration)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        # Calculating the accuracy of the model
        accuracy = accuracy_score(y_pred, y_test)

        # Print the accuracy of the model
        print(f"The model is {accuracy * 100}% accurate")
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=['Non tumor', 'Tumor']))


if __name__ == '__main__':

    brain_tumor = pd.read_csv(f'dataset/Brain Tumor.csv')
    class_brain_tumor = brain_tumor.loc[:, ['Class']]
    target = class_brain_tumor['Class'].values

    image_data = []
    for name in sorted(glob.glob('dataset/Brain Tumor/*'), key=len):
        im = cv2.imread(name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).flatten()
        image_data.append(im)
        pass

    (trainData, testData, trainLabels, testLabels) = train_test_split(image_data,
                                                                      target, test_size=0.25, random_state=42)

    # 10% of the training data used for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                    test_size=0.1, random_state=84)

    #knn(trainData, valData, trainLabels, valLabels, testLabels)
    #svm(trainData, testData, trainLabels, testLabels)


