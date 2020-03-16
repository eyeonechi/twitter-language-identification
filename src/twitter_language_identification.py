#==============================================================================
# COMP30027 Machine Learning
# 2017 Semester 1
# Project 2 - Short Text Language Identification
#
# Student Name  : Ivan Ken Weng Chee
# Student ID    : 736901
# Student Email : ichee@student.unimelb.edu.au
#
# project2.py
# Uses a trie-based classifier to predict the main language of text instances
#==============================================================================

# Library Imports
import codecs as cd
import operator as op
import pandas as pd
import random as rd

# Data Filenames
devData = 'data/Training/dev.json'
trainData = 'data/Training/train.json'
testData = 'data/Kaggle/test.json'
outputData = 'data/Output/output.csv'
errorData = 'data/Errors/errors.csv'

# Output Headers
errorHeader = "docid,predict,lang,uid\n"
outputHeader = "docid,lang\n"

#==============================================================================
# Data Preprocessing
#==============================================================================
""" Reads data from an input file into a Pandas DataFrame object """
def preprocessData(filename):
    with cd.open(filename, 'r', 'utf-8-sig') as file:
        data = pd.read_json(file, lines=True)
        file.close()
    data = data.fillna('')
    return data

#==============================================================================
# Data Splitting
#==============================================================================
""" Splits the data into train and test sets randomly based on the split """
def splitData(data, trainSplit):
    train = pd.Series()
    test = pd.Series()
    for row in range(len(data)):
        if rd.random() < trainSplit:
            train.append(data[row])
        else:
            test.append(data[row])
    return train, test

#==============================================================================
# Data Storage
#==============================================================================
""" Fills a dictionary trie with text """
def updateTrie(root, text):
    for word in text.lower().strip().split():
        node = root
        for letter in word:
            node = node.setdefault(letter, {})
        if '_end_' not in node:
            node['_end_'] = 0
        node['_end_'] += 1

""" Returns > 0 if an instance is in the trie, 0 otherwise """
def inTrie(root, text):
    node = root
    for letter in text.lower().strip():
        if letter not in node:
            return 0
        node = node[letter]
    if '_end_' in node:
        return node['_end_']
    return 0

""" Adds a new text instance to the corresponding language trie """
def updateLibrary(lib, lang, text):
    if lang not in lib:
        lib[lang] = dict()
    updateTrie(lib[lang], text)

""" Returns true if a given text instance is contained in the library """
def inLibrary(lib, lang, text):
    if lang in lib:
        return inTrie(lib[lang], text)
    return False

#==============================================================================
# Classifier Training
#==============================================================================
""" Learns the classifier by constructing language tries based on the data """
def trainClassifier(data):
    lib = dict()
    for i in range(len(data)):
        print(i)
        lang = data['lang'].iloc[i]
        text = data['text'].iloc[i]
        updateLibrary(lib, lang, text)
    return lib

#==============================================================================
# Classifier Testing
#==============================================================================
""" Predicts a class value for a data instance based on learned data """
def testClassifier(lib, data):
    predicts = []
    for i in range(len(data)):
        predicts.append(getClass(lib, data['text'].iloc[i]))
    return pd.Series.from_array(predicts)

""" Assigns scores for text appearing in respective language tries """
def getClass(lib, text):
    scores = []
    for lang in list(lib.keys()):
        scores.append([0, lang])
        for word in text.split():
            if inTrie(lib[lang], word) > 0:
                scores[-1][0] += 1
    scores.sort(key=op.itemgetter(0), reverse=True)
    return scores[0][1]

""" Returns the total distance metric between two sentences """
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

""" Returns the Levenshtein Distance between two strings """
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

#==============================================================================
# Error Analysis
#==============================================================================
""" Returns errors which the classifer makes over development data """
def errorAnalysis(lib, dev):
    errors = []
    for i in range(len(dev)):
        lang = dev['lang'].iloc[i]
        text = dev['text'].iloc[i]
        uid = dev['uid'].iloc[i]
        predict = getClass(lib, text)
        if lang != predict:
            errors.append([predict, lang, uid])
    print("Accuracy = {}%:" % str((len(dev) - len(errors)) / len(dev) * 100))
    return pd.DataFrame(errors)

#==============================================================================
# Data Postprocessing
#==============================================================================
""" Writes output data to a csv file"""
def writeOutput(filename, header, data):
    with cd.open(filename, 'w') as output:
        output.write(header)
        for i in range(len(data)):
            output.write("test%s,%s\n" % (str(i).zfill(4), data.iloc[i]))
        output.close()

#==============================================================================
# Driver Function
#==============================================================================
""" Driver function for the program """
if __name__ is "__main__":
    dev = preprocessData(devData)
    train = preprocessData(trainData)
    test = preprocessData(testData)
    
    lib = trainClassifier(train[['lang', 'text']])
    errors= errorAnalysis(lib, dev)
    writeOutput(errorData, errorHeader, errors)
    predicts = testClassifier(lib, test)
    writeOutput(outputData, outputHeader, predicts)
