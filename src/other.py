"""
COMP30027 Machine Learning
2017 Semester 1
Project 2 - Language Identification
Student Name  : Ivan Ken Weng Chee
Student ID    : 736901
Student Email : ichee@student.unimelb.edu.au
project2.py
"""

import codecs as cd
import operator as op
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skl

columnNames = [
    'displayname',
    'lang',
    'location',
    'text',
    'uid'
]
splitMethod = [
    'Train and Test Sets',
    'K-Fold Cross Validation',
    'Leave One Out Cross Validation',
    'Repeated Random Test-Train Splits'
]
dataVisualisation = [
    'Histogram',
    'DensityPlot',
    'Boxplot',
    'CorrelationMatrix',
    'ScatterplotMatrix'
]
featureSelection = [
    'UnivariateSelection',
    'RecursiveFeatureElimination',
    'PrincipleComponentAnalysis',
    'FeatureImportance'
]
algorithmTuning = [
    'GridSearch',
    'RandomSearch'
]
evaluationMetric = [
    'Accuracy',
    'LogarithmicLoss',
    'AreaUnderCurve',
    'ConfusionMatrix',
    'ClassificationReport',
    'MeanAbsoluteError',
    'MeanSquaredError',
    'R2'
]

#==============================================================================
# Data Preprocessing
#==============================================================================

def preprocessData(filename):
    with cd.open(filename, 'r', 'utf-8-sig') as file:
        data = pd.read_json(file, lines=True)
        #raw = []
        #for line in file:
        #    raw.append(pd.io.json.loads(line))
        file.close()
    #data = pd.io.json.json_normalize(raw)
    #data = pd.read_json()
    data = data.fillna('')
    return data

def rescaleData(data):
    array = data.values
    # Separate array into input and output components
    X = array[:,0:8]
    Y = array[:,8]
    scaler = skl.preprocessing.MinMaxScaler(feature_range=(0, 1))
    xRescaled = scaler.fit_transform(X)
    # Summarize transformed data
    np.set_printoptions(precision=3)
    print(xRescaled[0:5,:])
    return xRescaled, Y

def standardizeData(data):
    array = data.values
    # Separate array into input and output components
    X = array[:,0:8]
    Y = array[:,8]
    scaler = skl.preprocessing.StandardScaler().fit(X)
    xRescaled = scaler.transform(X)
    # Summarize transformed data
    np.set_printoptions(precision=3)
    print(xRescaled[0:5,:])
    return xRescaled, Y

def normalizeData(data):
    array = data.values
    # Separate array into input and output components
    X = array[:,0:8]
    Y = array[:,8]
    scaler = skl.preprocessing.Normalizer().fit(X)
    xRescaled = scaler.transform(X)
    # Summarize transformed data
    np.set_printoptions(precision=3)
    print(xRescaled[0:5,:])
    return xRescaled, Y

def binarizeData(data):
    array = data.values
    # Separate array into input and output components
    X = array[:,0:8]
    Y = array[:,8]
    scaler = skl.preprocessing.Binarizer(threshold=0.0).fit(X)
    xRescaled = scaler.transform(X)
    # Summarize transformed data
    np.set_printoptions(precision=3)
    print(xRescaled[0:5,:])
    return xRescaled, Y

"""
Descriptive Statistics
"""
def displayData(data):
    # Description
    print(data.describe)
    # Shape
    print(data.shape)
    # Types
    print(data.dtypes)
    # Class counts
    # print(data.groupby('class').size())
    # Correlation between attributes
    print(data.corr(method='pearson'))
    # Skew of univariate distributions
    print(data.skew())
    # Data
    # print(data)

#==============================================================================
# Feature Selection
#==============================================================================

def featureSelection(data, method, k=4, nAttributes=3, nComponents=3):
    if method is 'UnivariateSelection':
        return univariateSelection(data)
    elif method is 'RecursiveFeatureElimination':
        return recursiveFeatureElimination(data, nAttributes)
    elif method is 'PrincipleComponentAnalysis':
        return principleComponentAnalysis(data)
    elif method is 'FeatureImportance':
        return featureImportance(data)
    else:
        print('Please specify a feature selection method')

def univariateSelection(data, k=4):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    # Feature extraction
    test = skl.feature_selection.SelectKBest(score_func=skl.feature_selection.chi2, k=k)
    fit = test.fit(X, Y)
    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    # Summarize selected features
    features = fit.transform(X)
    print(features[0:5,:])

def recursiveFeatureElimination(data, nAttributes=3):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    # Feature extraction
    model = skl.linear_model.LogisticRegression()
    rfe = skl.feature_selection.RFE(model, nAttributes)
    fit = rfe.fit(X, Y)
    print('Num Features: %d') % fit.n_features_
    print('Selected Features: %s') % fit.support_
    print('Feature Ranking: %s') % fit.ranking_

def principleComponentAnalysis(data, nComponents=3):
    array = data.values
    X = array[:,0:8]
    # Feature extraction
    pca = skl.decomposition.PCA(n_components=nComponents)
    fit = pca.fit(X)
    # Summarize components
    print('Explained Variance: %s') % fit.explained_variance_ratio_
    print(fit.components_)

def featureImportance(data):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    # Feature Extraction
    model = skl.ensemble.ExtraTreesClassifier()
    model.fit(X, Y)
    # Display the relative importance of each attribute
    print(model.feature_importances_)

#==============================================================================
# Data Splitting
#==============================================================================

def splitData(data, method, k=10, testSplit=0.33):
    if method is 'Train and Test Sets':
        return holdout(data, testSplit)
    elif method is 'K-Fold Cross Validation':
        return kFoldCrossValidation(data, k)
    elif method is 'Leave One Out Cross Validation':
        return leaveOneOutCrossValidation(data, k)
    elif method is 'Repeated Random Test-Train Splits':
        return repeatedRandomHoldout(data, testSplit)
    else:
        print('Please specify a split method')

def holdout(data, testSplit=0.33):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    seed = 30027
    xTrain, xTest, yTrain, yTest = skl.model_selection.train_test_split(X, Y, test_size=testSplit, random_state=seed)
    model = skl.linear_model.LogisticRegression()
    model.fit(xTrain, yTrain)
    result = model.score(xTest, yTest)
    print("Accuracy: %.3f%%") % (result * 100.0)

def kFoldCrossValidation(data, k=10):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    seed = 30027
    kFold = skl.model_selection.KFold(n_splits=k, random_state=seed)
    model = skl.linear_model.LogisticRegression()
    results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold)
    print("Accuracy: %.3f%% (%.3f%%)") % (results.mean() * 100.0, results.std() * 100.0)

def leaveOneOutCrossValidation(data, k=10):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    loocv = skl.model_selection.LeaveOneOut()
    model = skl.linear_model.LogisticRegression()
    results = skl.model_selection.cross_val_score(model, X, Y, cv=loocv)
    print("Accuracy: %.3f%% (%.3f%%)") % (results.mean() * 100.0, results.std() * 100.0)

def repeatedRandomHoldout(data, k=10, testSplit=0.33):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    seed = 30027
    kFold = skl.model_selection.ShuffleSplit(n_splits=k, test_size=testSplit, random_state=seed)
    model = skl.linear_model.LogisticRegression()
    results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold)
    print("Accuracy: %.3f%% (%.3f%%)") % (results.mean() * 100.0, results.std() * 100.0)

#==============================================================================
# Data Visualisation
#==============================================================================

def visualiseData(data, plot):
    if plot is 'Histogram':
        return histogram(data)
    elif plot is 'DensityPlot':
        return densityPlot(data)
    elif plot is 'Boxplot':
        return boxPlot(data)
    elif plot is 'CorrelationMatrix':
        return correlationMatrix(data)
    elif plot is 'ScatterplotMatrix':
        return scatterplotMatrix(data)
    else:
        print('Please specify a data visualisation plot')
    """
    pd.options.display.mpl_style = 'default'
    # Feature Distributions
    data.boxplot()
    data.hist()
    # Feature-Class Relationships
    data.groupby('class').hist()
    data.groupby('class').plas.hist(alpha=0.4)
    # Feature-Feature Relationships
    pd.tools.plotting.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
    """

def histogram(data):
    data.hist()
    mpl.pyplot.show()

def densityPlot(data):
    data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    mpl.pyplot.show()

def boxPlot(data):
    data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    mpl.pyplot.show()

def correlationMatrix(data):
    correlations = data.corr()
    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columnNames)
    ax.set_yticklabels(columnNames)
    mpl.pyplot.show()

def scatterplotMatrix(data):
    pd.tools.plotting.scatter_matrix(data)
    mpl.pyplot.show()

#==============================================================================
# Linear Machine Learning Algorithms
#==============================================================================

def logisticRegression(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.linear_model.LogisticRegression()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def linearDiscriminantAnalysis(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.discriminant_analysis.LinearDiscriminantAnalysis()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def linearRegression(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:13]
    Y = array[:,13]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.linear_model.Ridge()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def ridgeRegression(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.linear_model.LinearRegression()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def lassoRegression(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.linear_model.Lasso()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def elasticNetRegression(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.linear_model.ElasticNet()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

#==============================================================================
# Nonlinear Machine Learning Algorithms
#==============================================================================

def kNearestNeighbor(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.neighbors.KNeighborsClassifier()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def naiveBayes(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.naive_bayes.GaussianNB()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def decisionTree(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:8]
    Y = array[:,8]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.tree.DecisionTreeClassifier()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)

def supportVectorMachine(data, metric, nSplits=10, seed=np.random.rand()):
    array = data.values
    X = array[:,0:13]
    Y = array[:,13]
    kFold = skl.model_selection.KFold(n_splits=nSplits, random_state=seed)
    model = skl.svm.SVC()
    return evaluationMetrics(X, Y, kFold, metric, model, seed)


def evaluationMetrics(X, Y, kFold, metric, model, testSplit=0.33, seed=np.random.rand()):
    if metric is 'Accuracy':
        results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold, scoring='accuracy')
        print('Accuracy: %.3f (%.3f)') % (results.mean(), results.std())
    elif metric is 'LogarithmicLoss':
        results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold, scoring='neg_log_loss')
        print('Logarithmic Loss: %.3f (%.3f)') % (results.mean(), results.std())
    elif metric is 'AreaUnderCurve':
        results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold, scoring='roc_auc')
        print('Area Under ROC Curve: %.3f (%.3f)') % (results.mean(), results.std())
    elif metric is 'ConfusionMatrix':
        xTrain, xTest, yTrain, yTest = skl.model_selection.train_test_split(X, Y, test_size=testSplit, random_state=seed)
        model.fit(xTrain, yTrain)
        predicted = model.predict(xTest)
        print(skl.metrics.confusion_matrix(yTest, predicted))
    elif metric is 'ClassificationReport':
        xTrain, xTest, yTrain, yTest = skl.model_selection.train_test_split(X, Y, test_size=testSplit, random_state=seed)
        model.fit(xTrain, yTrain)
        predicted = model.predict(xTest)
        print(skl.metrics.classification_report(yTest, predicted))
    elif metric is 'MeanAbsoluteError':
        results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold, scoring='neg_mean_absolute_error')
        print('Mean Absolute Error: %.3f (%.3f)') % (results.mean(), results.std())
    elif metric is 'MeanSquaredError':
        results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold, scoring='neg_mean_squared_error')
        print('Mean Squared Error: %.3f (%.3f)') % (results.mean(), results.std())
    elif metric is 'R2':
        results = skl.model_selection.cross_val_score(model, X, Y, cv=kFold, scoring='r2')
        print('R^2: %.3f (%.3f)') % (results.mean(), results.std())
    else:
        print('Please specify an evaluation metric')

#==============================================================================
# Algorithm Tuning
#==============================================================================

def algorithmTuning(data, method):
    if method is 'GridSearch':
        return gridSearch(data)
    elif method is 'RandomSearch':
        return randomSearch(data)
    else:
        print('Please specify an algorithm tuning method')

def gridSearch(data):
    # Prepare a range of alpha values to test
    alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    # Create and fit a ridge regression model, testing each alpha
    model = skl.linear_model.Ridge()
    grid = skl.model_selection.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    print(grid)
    # Summarize the results of the grid search
    print(grid.best_score_)
    print(grid.best_estimator_.alpha)

def randomSearch(data):
    # Prepare a uniform distribution to sample for the alpha parameter
    paramGrid = {'alpha': sp.stats.uniform()}
    # Create and fit a ridge regression model, testing random alpha values
    model = skl.linear_model.Ridge()
    rSearch = skl.model_selection.RandomizedSearchCV(estimator=model, param_distributions=paramGrid, n_iter=100)
    rSearch.fit(data.data, data.target)
    # Summarize the results of the random parameter search
    print(rSearch.best_score_)
    print(rSearch.best_estimator_.alpha)

def trainClassifier(data):
    lib = dict()
    for i in range(len(data)):
        lang = data['lang'].iloc[i]
        sentence = data['text'].iloc[i]
        updateLibrary(lib, lang, sentence)
#        for j in range(len(sentence)):
#            updateLibrary(lib, lang, sentence[j])
    return lib

def testClassifier(lib, data, gold=False):
    predict = []
    for i in range(len(data)):
        predict.append(getClass(lib, data['text'].iloc[i]))
    predict = pd.Series.from_array(predict)
    return predict

def writeOutput(filename, predict):
    with cd.open(filename, 'w') as output:
        output.write("docid,lang\n")
        for i in range(len(predict)):
            output.write("test%s,%s\n" % (str(i).zfill(4), predict.iloc[i]))
        output.write("\n")
        output.close()
#==============================================================================
# Data Storage
#==============================================================================

def updateTrie(root, text):
    for word in text.split():
        node = root
        for letter in word:
            node = node.setdefault(letter, {})
        if '_end_' not in node:
            node['_end_'] = 0
        node['_end_'] += 1
        

def inTrie(root, text):
    node = root
    for letter in text:
        if letter not in node:
            return 0
        node = node[letter]
    if '_end_' in node:
        return node['_end_']
    return 0

def updateLibrary(lib, lang, text):
    if lang not in lib:
        lib[lang] = dict()
    updateTrie(lib[lang], text)

def inLibrary(lib, lang, text):
    if lang in lib:
        return inTrie(lib[lang], text)
    return False

def getClass(lib, text):
    scores = []
    for lang in list(lib.keys()):
        scores.append([0, lang])
        for word in text.split():
            scores[-1][0] += inTrie(lib[lang], word)
#        for letter in text:
#            scores[-1][0] += inTrie(lib[lang], letter)
    scores.sort(key=op.itemgetter(0), reverse=True)
    return scores[0][1]

def compareSentences(s1, s2):
    score = 0
    s1 = s1.split(' ')
    s2 = s2.split(' ')
    for word in s1:
        word = word.lower()
    for word in s2:
        word = word.lower()
    for i in range(min(len(s1), len(s2))):
        distance = levenshteinDistance(s1[i], s2[i])
        score += distance
    return score

memo = {}
def levenshteinDistance(w1, w2):
    if w1 == "":
        return len(w2)
    if w2 == "":
        return len(w1)
    cost = 0 if w1[-1] == w2[-1] else 1
    i1 = (w1[:-1], w2)
    if not i1 in memo:
        memo[i1] = levenshteinDistance(*i1)
    i2 = (w1, w2[:-1])
    if not i2 in memo:
        memo[i2] = levenshteinDistance(*i2)
    i3 = (w1[:-1], w2[:-1])
    if not i3 in memo:
        memo[i3] = levenshteinDistance(*i3)
    return min([memo[i1] + 1, memo[i2] + 1, memo[i3] + cost])
      
def getText(lib, words):
    for k, v in lib.items():
        if isinstance(v, dict):
            getText(v, words)
        else:
            words.append(v)
    return set(words)

"""
Driver function
"""
def main():
    trainData = 'Kaggle/project2/dev.json'
    testData = 'Kaggle/project2/test.json'
    outputData = 'Kaggle/project2/output.csv'

    data = preprocessData(trainData)
    lib = trainClassifier(data[['lang', 'text']])
    test = preprocessData(testData)
    predict = testClassifier(lib, test)
    writeOutput(outputData, predict)

if __name__ is "__main__":
    main()
