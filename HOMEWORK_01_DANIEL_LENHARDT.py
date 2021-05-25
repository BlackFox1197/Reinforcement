


# LITTLE HOMEWORK:
# For the little homework you can choose between 2 tasks:
# a) Train and evaluate 3 scikit learn classifiers on a data set from openml.org
# OR
# b) Use my RNN trained on the IMDB data set used for sentiment classification (see above) and
# push the score (current=0.5, your goal should be at least 0.8 ACCURACY)

# AS A GUIDE YOU CAN FOLLOW THE STEPS BELOW:

#-----


from sklearn import datasets
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_surface(clf, X, y,
                 xlim=(-10, 10), ylim=(-10, 10), n_steps=250,
                 subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
                         np.linspace(ylim[0], ylim[1], n_steps))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    if show:
        plt.show()

## 1. DATA
# Find a data set that you like on: https://openml.org/ (if possible choose binary classification datasets because you can visualize them nicely)
# OR
# Use my IMDB data set from above




# I choose https://www.openml.org/d/1489 as data (2 classes 5 features)
input_file = "php8Mz7BG.csv"


# comma delimited is the default
data = pd.read_csv(input_file, header=0)
#data = np.loadtxt(input_file, delimiter=',')

## How does your data look like?
print(data)

y = data[['Class']].values.ravel()
#X = data[['V1', 'V2', 'V3', 'V4', 'V5']]
X = data[['V1', 'V4']].values

## How does a single data row look like?
print(X[0:1])


## How does your target vector look like?
print(y)
## 2. MODEL
## Choose 3(!) different models from: https://scikit-learn.org/stable/supervised_learning.html
# OR
# improve my RNN (e.g. using LSTM, BidirectionalLSTM, ....)
#
# labels = ["b", "r"]
# y = np.take(labels, (y < 10))
## 3. TRAINING
# # train your models on your training data
# # plot the classification surfaces or the learning curves (openml task)
# # OR
# # plot only the learning curves (IMDB task)

#SVM
from sklearn import svm
SVMclf = svm.SVC(probability=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

SVMclf.fit(X_train, y_train)
plot_surface(SVMclf, X_train, y_train)




#Random Forest
from sklearn.ensemble import RandomForestClassifier
ranForclf = RandomForestClassifier(n_estimators=500)
ranForclf.fit(X_train, y_train)

plot_surface(ranForclf, X_train, y_train)

#NN
from sklearn.neural_network import MLPClassifier
NNCLF = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation="relu", learning_rate="invscaling")
NNCLF.fit(X_train, y_train)
plot_surface(NNCLF, X_train, y_train)



from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    return plt

fig, axes = plt.subplots(3, 1, figsize=(10, 15))



title = "Learning Curves (Neural Networks)"
# Cross validation with 10 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(NNCLF, title, X, y, axes=axes[:,], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 15))



title = "RanForests"
# Cross validation with 10 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(ranForclf, title, X, y, axes=axes[:,], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()

title = "SVM"
# Cross validation with 10 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(SVMclf, title, X, y, axes=axes[:,], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()


## 4. EVALUATION
## Compute the LOSS
## Compute the ACCURACY
## Apply CROSS VALIDATION
## Compute a CONFUSION MATRIX
## Compute and visualize ROC AUC

from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y)

#print("Training error =", zero_one_loss(y_train, nearest_neighbor_clf.predict(X_train))) # DONT DO THAT!
print("Error/Loss =", zero_one_loss(y_test, SVMclf.predict(X_test)))
print("Accuracy =", accuracy_score(y_test, SVMclf.predict(X_test)))

## CROSS VALIDATION
# https://www.google.com/search?q=cross+validation&sxsrf=ALeKk03qO8p_xDhPvH2KIbrrM7srcM-Ihg:1618910334329&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiH6Zn7vozwAhWthv0HHdk5CkMQ_AUoAXoECAEQAw&biw=1920&bih=948#imgrc=FR44AeJ3c_W9nM
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
scores = []
scores = cross_val_score(svm.SVC(), X, y,
                         scoring="accuracy")
print("CV error = %f +-%f" % (1. - np.mean(scores), np.std(scores)))

# CONFUSION MATRIX
# https://www.google.com/search?q=confusion+matrix&sxsrf=ALeKk00LyNTJH3uVnt_Qj69nOwdbhaWs0A:1618909841676&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjt3KSQvYzwAhUHg_0HHWfiBa4Q_AUoAXoECAEQAw&biw=1920&bih=948#imgrc=d3aGGrkwCqhD9M
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, SVMclf.predict(X_test))

## ROC/AUC, Receiver Opertor Characteristic/Area Under Curve
# Definition: Area under the curve of the false positive rate (FPR) against the true positive rate (TPR) as the decision threshold of the classifier is varied.
# convert r to 1 and b to 0 ...just to be nice :)
y_train = (y_train == 1)
y_test = (y_test == 2)
from sklearn.metrics import get_scorer
roc_auc_scorer = get_scorer("roc_auc")
print("ROC AUC =", roc_auc_scorer(SVMclf, X_test, y_test))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, SVMclf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()









## 5. SEND your data to henrik.voigt@uni-jena.de by using the naming convention: HOMEWORK_01_FIRSTNAME_LASTNAME.ipynb until May 19th 00.00 AM/midnight






