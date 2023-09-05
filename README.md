# UNIVR-MachineLearning-Exam
This repo was created to support "Machine Learning &amp; Artificial Intelligence" exam.

In this code i'm using a [Brain Tumor dataset](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor) for trying some of the classical machine learning alghoritms. 
- PCA
- KNN with differents K
- PCA + KNN
- SVM with differents Kernels
- PCA + SVM
- A neural network

In order to run the code you need to download the dataset and extract into the code folder.
The codespace need to look like that:


    UNIVR-MachineLearning-Exam:
    |- dataset/
    |        |- Brain Tumor/ 
    |                     |- all the images are here
    |        |- Brain Tumor.csv
    |        |- bt_dataset_t3.csv
    |
    |- README.md
    |- .gitignore
    |- requirements.txt
    |- NeuralNetwork.py
    |- main.py
  
---------------------------------------------------
PCA
---------------------------------------------------
number of components = 100

![Alt text](plot/PCA100.png?raw=true "Title")

---------------------------------------------------
KNN
---------------------------------------------------

- k=1, accuracy=95.41%
- k=3, accuracy=95.76%
- k=5, accuracy=92.23%
- k=7, accuracy=91.17%
- k=9, accuracy=91.52%


- k=3 achieved highest accuracy of 95.76% on validation data

EVALUATION ON TESTING DATA

                precision    recall    f1-score   support

           0       0.95      0.94      0.95       526
           1       0.93      0.94      0.93       415

    accuracy                           0.94       941
    macro avg      0.94      0.94      0.94       941
    weighted avg   0.94      0.94      0.94       941


--------------------------------------------------
PCA + KNN 
---------------------------------------------------
number of components for PCA = 2

- k=1, accuracy=77.39%
- k=3, accuracy=80.92%
- k=5, accuracy=81.63%
- k=7, accuracy=81.63%
- k=9, accuracy=82.33%


- k=9 achieved highest accuracy of 82.33% on validation data

EVALUATION ON TESTING DATA

                precision    recall    f1-score   support

           0       0.82      0.83      0.82       526
           1       0.78      0.76      0.77       415

    accuracy                           0.80       941
    macro avg      0.80      0.80      0.80       941
    weighted avg   0.80      0.80      0.80       941

---------------------------------------------------
SVM
---------------------------------------------------

- Linear

  - The model is 87.67268862911796% accurate
  - Precision: 0.8405466970387244
  - Recall: 0.8891566265060241
              
                      precision   recall    f1-score   support

        Non tumor       0.91      0.87      0.89       526
        Tumor           0.84      0.89      0.86       415

        accuracy                            0.88       941
        macro avg       0.87      0.88      0.88       941
        weighted avg    0.88      0.88      0.88       941

- Poly 
  - The model is 90.54197662061637% accurate
  - Precision: 0.8790697674418605
  - Recall: 0.9108433734939759
  
                      precision   recall    f1-score   support

        Non tumor       0.93      0.90      0.91       526
        Tumor           0.88      0.91      0.89       415

        accuracy                            0.91       941
        macro avg       0.90      0.91      0.90       941
        weighted avg    0.91      0.91      0.91       941

- RBF 
  - The model is 90.43570669500531% accurate
  - Precision: 0.9012345679012346
  - Recall: 0.8795180722891566
              
                      precision   recall    f1-score   support

        Non tumor       0.91      0.92      0.92       526
        Tumor           0.90      0.88      0.89       415

        accuracy                            0.90       941
        macro avg       0.90      0.90      0.90       941
        weighted avg    0.90      0.90      0.90       941

---------------------------------------------------
SVM + PCA
---------------------------------------------------
number of components for PCA = 3

- EVALUATION ON VALIDATION DATA
- Linear 

  - The model is 36.74911660777385% accurate
  - Precision: 0.33986928104575165
  - Recall: 0.4
  
                     precision    recall    f1-score   support

        Non tumor       0.40      0.34      0.37       153
        Tumor           0.34      0.40      0.37       130

        accuracy                            0.37       283
        macro avg       0.37      0.37      0.37       283
        weighted avg    0.37      0.37      0.37       283

- Poly 

  - The model is 75.97173144876325% accurate
  - Precision: 0.6845238095238095
  - Recall: 0.8846153846153846
  
                   precision    recall    f1-score   support

        Non tumor     0.87      0.65      0.75       153
        Tumor         0.68      0.88      0.77       130

        accuracy                          0.76       283
        macro avg     0.78      0.77      0.76       283
        weighted avg  0.78      0.76      0.76       283

- RBF
  - The model is 84.45229681978799% accurate
  - Precision: 0.8307692307692308
  - Recall: 0.8307692307692308
  
                   precision    recall    f1-score   support

        Non tumor     0.86      0.86      0.86       153
        Tumor         0.83      0.83      0.83       130

        accuracy                          0.84       283
        macro avg     0.84      0.84      0.84       283
        weighted avg  0.84      0.84      0.84       283

