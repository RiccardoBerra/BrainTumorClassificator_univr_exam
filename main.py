import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import cv2
import glob
import plotly.express as px
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns
from sklearn import svm

from sklearn.decomposition import PCA
import time


def knn(x_train, x_test, x_valData, y_train, y_test, y_vallabels) -> None:

    kVals = range(1, 10, 2)
    accuracies = []
    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 10, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        # evaluate the model and update the accuracies list
        score = model.score(x_valData, y_vallabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))

    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("EVALUATION ON TESTING DATA")
    print(accuracy_score(y_test,predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test,predictions))
'''
    for i in list(map(int, np.random.randint(0, high=len(y_test), size=(1,)))):
        # grab the image and classify it
        image = x_test[i]
        prediction = model.predict(image.reshape(1, -1))[0]
        image = image.reshape((240, 240)).astype("uint8")

        # show the prediction
        print("Has Brain tumor: {}  Test Labels : {}".format(prediction, y_test[i]))
        cv2.imshow("Image ", image)
        cv2.waitKey(0)

'''
def knn_with_pca(x_train, x_test, x_valData, y_train, y_test, y_vallabels, n_components) -> None:
    # Addestra la PCA sui dati di addestramento
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    # Applica la PCA ai dati di addestramento, validazione e test
    trainData_pca = pca.transform(x_train)
    valData_pca = pca.transform(x_valData)
    testData_pca = pca.transform(x_test)

    kVals = range(1, 10, 2)
    accuracies = []

    # Loop su vari valori di k per il classificatore KNN
    for k in range(1, 10, 2):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData_pca, y_train)
        score = model.score(valData_pca, y_vallabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # Trova il valore di k che ha la massima accuratezza
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData_pca, y_train)
    predictions = model.predict(testData_pca)

    print("EVALUATION ON TESTING DATA")
    print(accuracy_score(y_test,predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test,predictions))
'''
    # Plot the data points in 2D PCA space
    plt.figure(figsize=(10, 6))
    plt.scatter(trainData_pca[:, 0], trainData_pca[:, 1], c=y_train, cmap='viridis', label='Train Data')
    plt.scatter(testData_pca[:, 0], testData_pca[:, 1], c=y_test, cmap='viridis', marker='x', label='Test Data')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Components 1 vs 2')
    plt.legend(loc='best')
    plt.show()


    for i in list(map(int, np.random.randint(0, high=len(y_test), size=(50,)))):
        #model want as input only testdata as same size as PCA transformation
        image = testData_pca[i]
        prediction = model.predict(image.reshape(1, -1))[0]

        print("Has Brain tumor: {}  Test Labels : {}".format(prediction, y_test[i]))

'''
def svm(x_train, x_test, y_train, y_test) -> None:
    # Creating a support vector classifier
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    max_iteration = 100


    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        start_time2 = time.time()

        from sklearn import svm
        print('----', kernel, '----')
        model = svm.SVC(kernel=kernel, max_iter=max_iteration)
        model.fit(x_train, y_train)
        '''
        y_pred = model.predict(valData_pca)
        # Calculating the accuracy of the model  ------- VALIDATION DATA
        accuracy = accuracy_score(y_pred, y_vallabels)
        # Print the accuracy of the model
        print("EVALUATION ON VALIDATION DATA")
        print(f"The model is {accuracy * 100}% accurate")
        print("Precision:", precision_score(y_vallabels, y_pred))
        print("Recall:", recall_score(y_vallabels, y_pred))
        print(classification_report(y_vallabels, y_pred, target_names=['Non tumor', 'Tumor']))
        '''
        y_pred = model.predict(x_test)
        # Calculating the accuracy of the model  ------- VALIDATION DATA
        accuracy = accuracy_score(y_pred, y_test)
        # Print the accuracy of the model
        print("EVALUATION ON VALIDATION DATA")
        print(f"The model is {accuracy * 100}% accurate")
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=['Non tumor', 'Tumor']))
        print(confusion_matrix(y_test, y_pred))

        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"Elapsed time on {kernel}: ", elapsed_time2)
def svm_with_pca(x_train, x_test, x_valData, y_train, y_test, y_vallabels, n_components) -> None:
    pca = PCA(n_components=n_components)
    pca.fit(x_train)

    trainData_pca = pca.transform(x_train)
    valData_pca = pca.transform(x_valData)
    testData_pca = pca.transform(x_test)

    # Creating a support vector classifier
    max_iteration = 10000

    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        start_time2 = time.time()

        from sklearn import svm
        print('----', kernel, '----')
        model = svm.SVC(kernel=kernel, max_iter=max_iteration)
        model.fit(trainData_pca, y_train)
        '''
        y_pred = model.predict(valData_pca)
        # Calculating the accuracy of the model  ------- VALIDATION DATA
        accuracy = accuracy_score(y_pred, y_vallabels)
        # Print the accuracy of the model
        print("EVALUATION ON VALIDATION DATA")
        print(f"The model is {accuracy * 100}% accurate")
        print("Precision:", precision_score(y_vallabels, y_pred))
        print("Recall:", recall_score(y_vallabels, y_pred))
        print(classification_report(y_vallabels, y_pred, target_names=['Non tumor', 'Tumor']))
        '''
        y_pred = model.predict(testData_pca)
        # Calculating the accuracy of the model  ------- VALIDATION DATA
        accuracy = accuracy_score(y_pred, y_test)
        # Print the accuracy of the model
        print("EVALUATION ON VALIDATION DATA")
        print(f"The model is {accuracy * 100}% accurate")
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=['Non tumor', 'Tumor']))
        print(confusion_matrix(y_test, y_pred))

        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"Elapsed time on {kernel}: ", elapsed_time2)


def pca(x_train,x_test, y_train, y_test, n_components) -> None:
    pca = PCA(n_components)  # Scegli il numero di componenti principali da visualizzare
    x_train_pca = pca.fit_transform(x_train)  # Applica la PCA ai dati di addestramento
    x_test_pca = pca.transform(x_test)  # Applica la PCA ai dati di test
    print(x_train_pca.shape)
    print(x_test_pca.shape)
    '''
    # Plotta i dati di addestramento
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette='Set1', s=100)
    plt.title('PCA Plot of Training Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Class', labels=['Non tumor', 'Tumor'])
    plt.show()

    # Plotta i dati di test
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_test_pca[:, 0], y=x_test_pca[:, 1], hue=y_test, palette='Set1', s=100)
    plt.title('PCA Plot of Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Class', labels=['Non tumor', 'Tumor'])
    plt.show()

    '''
if __name__ == '__main__':

    brain_tumor = pd.read_csv(f'dataset/Brain Tumor.csv')
    class_brain_tumor = brain_tumor.loc[:, ['Class']]
    target = class_brain_tumor['Class'].values
    image_data = []

    for name in sorted(glob.glob('dataset/Brain Tumor/*'), key=len):
        im = cv2.imread(name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).flatten()
        # im = [im.mean(), im.std()]
        image_data.append(im)
        pass

    (x_train, x_test, y_train, y_test) = train_test_split(image_data,
                                                                      target, test_size=0.25, random_state=42)

    # 10% of the training data used for validation
    (x_train, x_valData, y_train, y_vallabels) = train_test_split(x_train, y_train,
                                                                    test_size=0.1, random_state=84)

    # Start timer
    start_time = time.time()

    knn(x_train, x_test, x_valData, y_train, y_test, y_vallabels)
    svm(x_train, x_test, y_train, y_test)
    pca(x_train, x_valData, y_train, y_vallabels, n_components=100)
    knn_with_pca(x_train, x_test, x_valData, y_train, y_test, y_vallabels, n_components=2)
    svm_with_pca(x_train, x_test, x_valData, y_train, y_test, y_vallabels, n_components=100)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)



