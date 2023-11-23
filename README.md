# Fashion-MNIST-Classification 
## Use Main_Code.ipynb for Project evaluation

Title: Comparative Analysis of Machine Learning and Deep Learning Models for Classification of the Fashion MNIST Dataset

The Fashion MNIST dataset, characterized by its diverse set of 28x28 grayscale images representing various fashion products, serves as an intricate benchmark for evaluating machine learning algorithms. This research, conducted as part of the ECEN 758 Data Mining and Analysis group project, aims to provide a comprehensive comparison of several classification models applied to the Fashion MNIST dataset. The selected models encompass a range of methodologies, including traditional machine learning approaches such as Support Vector Machine Classifier (SVM), XGBoost, K-Nearest Neighbours (KNN), Logistic Regression, Decision Tree Classifier, and Random Forest Classifier, as well as deep learning techniques such as Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The study focuses on assessing the efficacy of these models in image classification tasks through the meticulous evaluation of performance metrics, including accuracy, precision, recall, and F1-score.

Project Group Members:
1. Saptarshi Mondal
2. Kirthan Prakash
3. Kiran Babu Athina

Instructions:

This GitHub repo consists of below items:
1. Main_Code.ipynb - This is the primary code to load the dataset, perform EDA, train ML and DL models and test the models. We have taken only the best hyperparameters from FashionMNIST_MLmodels.ipynb notebook below to build our ML models. For DL architectures we have reduecd the epochs to reduce computing time (for project evaluation purposes), however we have saved the best model from FashionMNIST_ANN_CNN.ipynb to show the best accuracy.

2. FashionMNIST_MLmodels.ipynb - This notebook houses end to end implementation of all ML models (mentioned below) along with hyperparameter tuning. We have employed GridCV resulting in high computing time which are as follows:
Approximate Computing Time:
- KNN (K-Nearest Neighbors) - 12 min
- Logistic Regression - 2 min
- Decision Tree Classifier - 6 min
- Random Tree Classfier - 23 min

3. FashionMNIST_ANN_CNN.ipynb - This houses end to end implementation of ANN and CNN architecture trained for 50 epochs with patience of 10 epoch. 
Approximate Computing Time:
- ANN (on GPU) - 6 mins
- CNN (on GPU) -  1 hour

4. SVC_training.ipynb - This notebook houses end to end implementation of SVM Classifier (mentioned below) along with hyperparameter tuning. We have employed RandomizedSearchCV resulting in a high computing time of 1.5 days

5. XGBoost_training.ipynb - This notebook houses end to end implementation of SVM Classifier (mentioned below) along with hyperparameter tuning. We have employed RandomizedSearchCV resulting in an approximate computing time of 30 mins.

6. Plots - Navigate to this folder to view Loss Vs Epoch and Confusion Matrix plots for ANN and CNN. Please note, the plots for reduced epochs are named as 'plotname_countofepoch.jpg' ex Confusion_Matrix_ANN_3.jpg, plots with higher epochs view 'plotname.jpg'

7. Best_Models - We have saved best_models as checkpoints for ANN and CNN architecture for higher epochs for each fold in here.

8. Eval_Models - Same as Best_Models but trained with reduced epochs as in Main_Code.ipynb.










