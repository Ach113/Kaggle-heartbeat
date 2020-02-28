# Kaggle-heartbeat
Classification of ECG dataset found on Kaggle using Keras convolutional neural networks

Model is saved after training and then reconstructed using self-made functions. Model is simplified in order to avoid overly complicated structure, as it is intended for lower-level implementation.
# Data:
![ecg_data](https://i.imgur.com/VSMzysq.png)

# Models performance:
![Confusion_Matrix](https://i.imgur.com/KvFeV85.png)

              precision    recall  f1-score   support

           0       0.96      0.99      0.98     18117
           1       0.93      0.49      0.64       556
           2       0.89      0.79      0.84      1448
           3       0.70      0.50      0.58       162
           4       0.99      0.89      0.94      1608

    accuracy                           0.96     21891
    macro avg      0.90      0.73      0.80     21891
    weighted avg   0.95      0.96      0.95     21891
