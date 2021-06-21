#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION
# The task is to use Machine Learning algorithms to predict whether a person has diabetes or not. The outcome is based on the information about the patient such as the number of Pregnancies the patient had, the Glucose level, the Blood Pressure, the Skin Thickness, the	Insulin level, BMI, Diabetes Pedigree Function and Age. 

# In[93]:


#import the required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import *
style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix, f1_score, matthews_corrcoef


# For feature creation
# Degree 2 is used here but one can set the degree to be a hyperparameter to further explore the accuracy of the model
#poly = PolynomialFeatures(degree = 2)

#importing the classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

#for combining multiple models into one 
from sklearn.ensemble import StackingClassifier


# In[67]:


#read the dataset 
#the data was saved on my computer, so you can edit the file path to suit you
#this could be a url

diabetes_data = pd.read_csv("/Users/apple/Documents/OPEN_UNIVERSITY/Data_for_DataScience/Diabetes/diabetes-dataset.csv")


# In[4]:


#set what the datasets look like
print(diabetes_data.shape)
diabetes_data.head()


# In[5]:


# info about the datasets: to check if the dataset has null values
print(diabetes_data.info())
print(diabetes_data.isnull().sum())


# # Exploratory Analysis
# Some analysis of the data is done to check the dependence of the feature variables to the label variable

# In[6]:


#checking dependencies
plt.rcParams['figure.figsize'] = (20.0, 20.0)
fig, ax = plt.subplots(nrows = 3, ncols = 3)

diabetes_data['Pregnancies'].plot.hist(ax = ax[0,0], title = 'Pregnancies')
ax[0,0].yaxis.set_minor_locator(MultipleLocator(20))
ax[0,0].xaxis.set_minor_locator(MultipleLocator(0.5))

diabetes_data["Glucose"].plot.hist(ax = ax[0,1], title = 'Glucose level')
ax[0,1].yaxis.set_minor_locator(MultipleLocator(20))
ax[0,1].xaxis.set_minor_locator(MultipleLocator(10))

diabetes_data['BloodPressure'].plot.hist(ax = ax[0,2], title = 'Blood Pressure')
ax[0,2].yaxis.set_minor_locator(MultipleLocator(20))
ax[0,2].xaxis.set_minor_locator(MultipleLocator(4))

diabetes_data['SkinThickness'].plot.hist(ax = ax[1,0], title = 'Skin Thickness')
ax[1,0].yaxis.set_minor_locator(MultipleLocator(20))
ax[1,0].xaxis.set_minor_locator(MultipleLocator(4))

diabetes_data['Insulin'].plot.hist(ax = ax[1,1], title = 'Insulin')
ax[1,1].yaxis.set_minor_locator(MultipleLocator(40))
ax[1,1].xaxis.set_minor_locator(MultipleLocator(40))

diabetes_data['BMI'].plot.hist(ax = ax[1,2], title = 'BMI')
ax[1,2].yaxis.set_minor_locator(MultipleLocator(20))
ax[1,2].xaxis.set_minor_locator(MultipleLocator(4))

diabetes_data['DiabetesPedigreeFunction'].plot.hist(ax = ax[2,0], title = 'Diabetes Pedigree Function')
ax[2,0].yaxis.set_minor_locator(MultipleLocator(20))
ax[2,0].xaxis.set_minor_locator(MultipleLocator(0.1))

diabetes_data['Age'].plot.hist(ax = ax[2,1], title = 'Age')
ax[2,1].yaxis.set_minor_locator(MultipleLocator(20))
ax[2,1].xaxis.set_minor_locator(MultipleLocator(2))

sns.countplot(ax = ax[2,2], x = "Outcome", data = diabetes_data)

plt.show()


# # Data Splitting

# In[7]:


#split the data into features and labels
x = diabetes_data.drop(["Outcome"], axis = 1)
y = diabetes_data["Outcome"]

#split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)


# In[8]:


#scale the data due to large range of of the distribution
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Calling the models

# In[61]:


#Keep all the classifiers in a list so that the testing and training can be done once and for all
#then one can choose the one with the best accuracy
classifiers_ = [
    ("AdaBoost", AdaBoostClassifier()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Linear SVM", SVC(kernel="linear", C=1,probability=True)),
    ("Naive Bayes",GaussianNB()),
    ("Nearest Neighbors",KNeighborsClassifier(2)),
    ("Neural Net",MLPClassifier(alpha=1)),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1)),
    ("RBF SVM",SVC(gamma=2, C=1,probability=True)),
    ("SGDClassifier", SGDClassifier(max_iter=1000, tol=10e-3,penalty='elasticnet')),
    ("LogisticRegression", LogisticRegression()), 
    ("Perceptron", Perceptron(tol=1e-3, random_state=0)), 
    ("BaggingClassifier", BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0))
    ] 


# In[78]:


plt.rcParams['figure.figsize'] = (5.0, 5.0)
#use each Classifier to take its training results.
clf_names = []
train_accuracy_score = []
test_accuracy_score = []
predict_sums = []
test_f1score = []
train_f1score = []
test_matthews = []
train_matthews = []
i = 0

for n,clf in classifiers_:
    clf_names.append(n)
    
    # Model training
    clf.fit(X_train, y_train)
    print(i, ":",  n+" training done! \n ")
    
    # The prediction for both training and testing 
    clf.predict(X_test)
    clf.predict(X_train)
    predict_sums.append(clf.predict(X_test).sum()) 
        #you can print the classification report and confusion matrix if you like
    print(classification_report(y_test, clf.predict(X_test)))
    print(confusion_matrix(y_test, clf.predict(X_test)))
    
    #you can also plot the confussion matrix if you like
    disp1 = plot_confusion_matrix(clf, X_train, y_train,
                              display_labels=['YES','NO'],
                              cmap=plt.cm.Blues,
                              normalize=None)
    disp1.ax_.set_title('Confusion matrix')
    plt.show()
    
    disp = plot_confusion_matrix(clf, X_test, y_test,
                              display_labels=['YES','NO'],
                              cmap=plt.cm.Blues,
                              normalize=None)
    disp.ax_.set_title('Confusion matrix')
    plt.show()
    
    # Measure training accuracy and score
    train_accuracy_score.append(round(accuracy_score(y_train, clf.predict(X_train)), 3))
    train_matthews.append(round(matthews_corrcoef(y_train, clf.predict(X_train)),3))
    train_f1score.append(round(f1_score(y_train, clf.predict(X_train)),3))
    print("The Training Accuracy Score: ", accuracy_score(y_train, clf.predict(X_train)) )
    print("The Training F1 Score: ", f1_score(y_train, clf.predict(X_train)) )
    print("The Training Matthews coefficient: ", matthews_corrcoef(y_train, clf.predict(X_train)))
    print(n+" training score done!")
    
    # Measure test accuracy and score
    test_accuracy_score.append(round(accuracy_score(y_test, clf.predict(X_test)), 3))
    test_f1score.append(round(f1_score(y_test, clf.predict(X_test)),3))
    test_matthews.append(round(matthews_corrcoef(y_test, clf.predict(X_test)),3))
    print("The Accuracy Score: ", accuracy_score(y_test, clf.predict(X_test)))
    print("Test F1 Score: ",f1_score(y_test,clf.predict(X_test)))
    print("The Testing Matthews coefficient: ", matthews_corrcoef(y_test, clf.predict(X_test)))
    print(n+" testing score done!")
    print("-------------------------------------------------------")
    print("  ")
    i = i+1
print("Names: ", clf_names)
print("Train Accuracy Scores: ", train_accuracy_score)
print("Test Accuracy Scores: ", test_accuracy_score)
print("Train F1 Scores: ", accuracy_scores)
print("Test F1 Scores: ", test_f1score)
print("Train Matthews Coefficients", train_matthews)
print("Test Matthews Coefficients", test_matthews)


# In[80]:


plt.rcParams['figure.figsize'] = (40.0, 30.0)
figs, ax = plt.subplots(2,3)

ax[0,0].scatter(x =  train_accuracy_score,y = clf_names)
ax[0,0].set_title("Train Accuracy")
ax[0,0].grid(True, color = 'g')

ax[0,1].scatter(x =  train_f1score,y = clf_names)
ax[0,1].set_title("Train Accuracy")
ax[0,1].grid(True, color = 'g')

ax[0,2].scatter(x =  train_matthews,y = clf_names)
ax[0,2].set_title("Train Matthews Coefficients")
ax[0,2].grid(True, color = 'g')


ax[1,0].scatter(test_accuracy_score,clf_names)
ax[1,0].set_title("Test Accuracy")
ax[1,0].grid(True, color = 'g')


ax[1,1].scatter(test_f1score,clf_names)
ax[1,1].set_title("Test F1 Score")
ax[1,1].grid(True, color = 'g')

ax[1,2].scatter(test_matthews,clf_names)
ax[1,2].set_title("Test Matthews Coefficients")
ax[1,2].grid(True, color = 'g')

plt.show()


# From the graphs above, one would notice overfitting on the Random Forest classifier and Gaussian process but they also showed the best test accuracy. This is followed by the Radial Basis Function (RBF) kernel of the Support Vector Machine (SVM), Decision Tree and Nearest Neighbors.  

# # Polynomial Features

# In[35]:


#The PolynomialFeatures will be used for feature creation to explore the nonlinear pattern of the numerical data.
from sklearn.preprocessing import PolynomialFeatures
#The Pipeline is used to package the feature creator and the classifier.
from sklearn.pipeline import Pipeline


# For feature creation
# Degree 2 is used here but one can set the degree to be a hyperparameter to further explore the accuracy of the model
poly = PolynomialFeatures(degree = 2)


# In[74]:


plt.rcParams['figure.figsize'] = (5.0, 5.0)
#use each Classifier to take its training results.
clf_names = []
train_accuracy_score = []
test_accuracy_score = []
predict_sums = []
test_f1score = []
train_f1score = []
test_matthews = []
train_matthews = []
i = 0

for n,clf in classifiers_:
    clf_names.append(n)
    
    # Model declaration with pipeline
    clf = Pipeline([('POLY', poly),('CLF',clf)])
    
    # Model training
    clf.fit(X_train, y_train)
    print(i, ":",  n+" training done! \n ")
    
    # The prediction for both training and testing 
    clf.predict(X_test)
    clf.predict(X_train)
    predict_sums.append(clf.predict(X_test).sum()) 
        #you can print the classification report and confusion matrix if you like
    print(classification_report(y_test, clf.predict(X_test)))
    print(confusion_matrix(y_test, clf.predict(X_test)))
    
    #you can also plot the confussion matrix if you like
    disp1 = plot_confusion_matrix(clf, X_train, y_train,
                              display_labels=['YES','NO'],
                              cmap=plt.cm.Blues,
                              normalize=None)
    disp1.ax_.set_title('Confusion matrix')
    plt.show()
    
    disp = plot_confusion_matrix(clf, X_test, y_test,
                              display_labels=['YES','NO'],
                              cmap=plt.cm.Blues,
                              normalize=None)
    disp.ax_.set_title('Confusion matrix')
    plt.show()
    
    # Measure training accuracy
    train_accuracy_score.append(round(accuracy_score(y_train, clf.predict(X_train)), 3))
    train_matthews.append(round(matthews_corrcoef(y_train, clf.predict(X_train)),3))
    train_f1score.append(round(f1_score(y_train, clf.predict(X_train)),3))
    print("The Training Accuracy Score: ", accuracy_score(y_train, clf.predict(X_train)) )
    print("The Training F1 Score: ", f1_score(y_train, clf.predict(X_train)) )
    print("The Training Matthews coefficient: ", matthews_corrcoef(y_train, clf.predict(X_train)))
    print(n+" training score done!")
    
    # Measure test accuracy 
    test_accuracy_score.append(round(accuracy_score(y_test, clf.predict(X_test)), 3))
    test_f1score.append(round(f1_score(y_test, clf.predict(X_test)),3))
    test_matthews.append(round(matthews_corrcoef(y_test, clf.predict(X_test)),3))
    print("The Accuracy Score: ", accuracy_score(y_test, clf.predict(X_test)))
    print("Test F1 Score: ",f1_score(y_test,clf.predict(X_test)))
    print("The Testing Matthews coefficient: ", matthews_corrcoef(y_test, clf.predict(X_test)))
    print(n+" testing score done!")
    print("-------------------------------------------------------")
    print("  ")
    i = i+1
print("Names: ", clf_names)
print("Train Accuracy Scores: ", train_accuracy_score)
print("Test Accuracy Scores: ", test_accuracy_score)
print("Train F1 Scores: ", accuracy_scores)
print("Test F1 Scores: ", test_f1score)
print("Train Matthews Coefficients", train_matthews)
print("Test Matthews Coefficients", test_matthews)


# In[77]:


plt.rcParams['figure.figsize'] = (30.0, 25.0)
figs, ax = plt.subplots(2,3)

ax[0,0].scatter(x =  train_accuracy_score,y = clf_names)
ax[0,0].set_title("Train Accuracy")
ax[0,0].grid(True, color = 'g')

ax[0,1].scatter(x =  train_f1score,y = clf_names)
ax[0,1].set_title("Train Accuracy")
ax[0,1].grid(True, color = 'g')

ax[0,2].scatter(x =  train_matthews,y = clf_names)
ax[0,2].set_title("Train Matthews Coefficients")
ax[0,2].grid(True, color = 'g')


ax[1,0].scatter(test_accuracy_score,clf_names)
ax[1,0].set_title("Test Accuracy")
ax[1,0].grid(True, color = 'g')


ax[1,1].scatter(test_f1score,clf_names)
ax[1,1].set_title("Test F1 Score")
ax[1,1].grid(True, color = 'g')

ax[1,2].scatter(test_matthews,clf_names)
ax[1,2].set_title("Test Matthews Coefficients")
ax[1,2].grid(True, color = 'g')

plt.show()


# When compared to the linear input features, although some of the models performed better with polynomial input features, the ones with the best accuracies did not improve. 
# 
# One can combine the few best multiple models into a single model to obtain a hopefully better accuracy. The models are combined using the Stacking Classifier. That is, combining the prediction probabilities from multiple machine learning models on the same dataset. 

# In[90]:


#THE STACKING CLASSIFIER
#Let's use the combination of the prediction probability of the best five classifiers

bestclassifiers = [
    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1)),
    ("RBF SVM",SVC(gamma=2, C=1,probability=True)),
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
    ("Nearest Neighbors",KNeighborsClassifier(2))
                  ] 

#Build the stack model
stack_model = StackingClassifier( estimators = bestclassifiers, final_estimator = RandomForestClassifier(n_jobs=2, random_state=1) )
#I used the best estimator as my final estimator to optimise the result.

#train the stack model
stack_model.fit(X_train, y_train)
    
# The prediction for both training and testing 
stack_model.predict(X_test)
stack_model.predict(X_train)

# Measure training accuracy

print("The Training Accuracy Score: ", accuracy_score(y_train, stack_model.predict(X_train)) )
print("The Training F1 Score: ", f1_score(y_train, stack_model.predict(X_train)) )
print("The Training Matthews coefficient: ", matthews_corrcoef(y_train, stack_model.predict(X_train)))
print(n+" training score done!")
    
# Measure test accuracy 

print("The Accuracy Score: ", accuracy_score(y_test, stack_model.predict(X_test)))
print("Test F1 Score: ",f1_score(y_test,stack_model.predict(X_test)))
print("The Testing Matthews coefficient: ", matthews_corrcoef(y_test, stack_model.predict(X_test)))


# Although the stack_model (with accuracy score = 0.97, f1 score = 0.958 and Matthews coefficient = 0.934) performed way better than most of the models used, it is still not as accurate as the Random Forest Classifier (with accuracy Score = 0.978 , f1 score = 0.969 and Matthews coefficient = 0.952). So, I would rather stick with the single model when deploying.  

# # Most important feature

# In[31]:


#to obtain the feature importance 
plt.rcParams['figure.figsize'] = (12.0, 5.0) 

theclassifiers = classifiers_ = [
    ("AdaBoost", AdaBoostClassifier()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
    ("Random Forest",RandomForestClassifier(n_jobs=2, random_state=1))
    ]  
#NB: these classifiers are considered because they are only once amongst the classfiers considered that run with the 
# fitted attribute "feature_importances_"



for n,clif in theclassifiers:
    clif.fit(X_train, y_train)
#put the name of the column heads in a list
    column_head = diabetes_data.drop(["Outcome"], axis = 1).columns

#obtain the coefficients of the features
    coefs = clif.feature_importances_.flatten()
    
#Make a bar chart of the coefficients
    sns.barplot(x=column_head[0:len(column_head)],
                y=coefs[0:len(column_head)])
    plt.title(n, fontsize=25)
    plt.ylabel("Coefficients", fontsize=22)
    plt.xlabel("Feature Name", fontsize=22)
    plt.grid(True, color = "g")
    plt.show()


# In[48]:


#to obtain the feature importance 
plt.rcParams['figure.figsize'] = (12.0, 5.0)

theclassifiers = [   
    ("Linear SVM", SVC(kernel="linear", C=1,probability=True)),
    ("Perceptron", Perceptron(tol=1e-3, random_state=0))
    ] 
#NB: these classifiers are considered because they are only once amongst the classfiers considered that run with the 
# fitted attribute "coef_"

for n,clif in theclassifiers:
    clif.fit(X_train, y_train)

#put the name of the column heads in a list
    column_head = diabetes_data.drop(["Outcome"], axis = 1).columns

#obtain the coefficients of the features
    coefs = clif.coef_.flatten()
    
#Make a bar chart of the coefficients
    sns.barplot(x=column_head,
                y=coefs)
    plt.title(n, fontsize=25)
    plt.ylabel("Coefficients", fontsize=22)
    plt.xlabel("Feature Name", fontsize=22)
    plt.grid(True, color = "g")
    plt.show()


# The graphs above show varying values of importance for the features (as expected) depending on the model deployed. The non-zero coefficients values suggest that all the features are important. All the features are essentially used in the model such that removing anyone would affect the accuracy of the model. 

# # Deploying the model

# Since we have found out that the best model is the RandomForestClassifier. This model is typically deployed by exporting the model and binding it with an application API. Here, I will just try to take the details of the from the patient to predict if the patient has diabetes or not. 

# In[94]:


#PREGNANCIES
pregnancies = input("How many preganancies has the patient had:\n ")
pregnancies  = int(pregnancies)

#GLUCOSE LEVEL
glucose_level = input("Glucose Level: \n ")
glucose_level = float(glucose_level)

#BLOOD PRESSURE
blood_pressure = input("Blood Pressure: \n ")
blood_pressure = float(blood_pressure)
    
#Skin Thickness
skin_thickness = input("Skin Thickness:\n ")
skin_thickness = float(skin_thickness)

#Insulin level
insulin_level = input("Insulin level:\n ")
insulin_level = float(insulin_level)

#Body mass index (BMI) 
bmi = input("Body mass index (BMI):  \n ")
bmi = float(bmi)

#Diabetes Pedigree Function 
diabetes_ped  = input("Diabetes Pedigree Function: \n")
diabetes_ped = float(diabetes_ped)

#Age
age = input("Age: \n")
age = int(age)

1

    
#The prediction based on the details
print("\n\n THE DIABETES PREDICTION: ")
Xnew = [[pregnancies, glucose_level, blood_pressure, skin_thickness, insulin_level, bmi, diabetes_ped, age]]
Xnew = sc.transform(Xnew)

RandomForestCF = RandomForestClassifier(n_jobs=2, random_state=1)
RandomForestCF.fit(X_train, y_train)

ynew = RandomForestCF.predict(Xnew)
if ynew[0]==0: 
    print("No diabetes detected!!!")
elif ynew[0]==1:
    print("Diabetes detected!!!") 

