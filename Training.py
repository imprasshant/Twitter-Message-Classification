import pandas as pd
import numpy
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import matplotlib.pyplot as plot
plot.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import plot_model


import xlrd
new_book = xlrd.open_workbook('Randombook.xls')   # Reading data from excel file
sheet_to_load = new_book.sheet_by_name('Random_data')


csv_file_train = open('Randombook_csv_train.csv', 'wb')   # Open file in write mode for training data
wr_train = csv.writer(csv_file_train, quoting=csv.QUOTE_ALL)

csv_file_test = open('Randombook_csv_test.csv', 'wb')      # Open file in write mode for test data
wr_test = csv.writer(csv_file_test, quoting=csv.QUOTE_ALL)

rowno=sheet_to_load.nrows
wr_test.writerow(sheet_to_load.row_values(0))               # Copying the first row of excel file which contain column name of each field

for rownum in xrange(rowno):
    if rownum<=(.7*rowno):
        wr_train.writerow(sheet_to_load.row_values(rownum))   # Distributing the whole data into train and test data
    else:
        wr_test.writerow(sheet_to_load.row_values(rownum))

csv_file_train.close()
csv_file_test.close()

train_data = pd.read_csv("Randombook_csv_train.csv", header=0,delimiter=",",quotechar='"', quoting=csv.QUOTE_ALL,names=['tweet','topic']) # Read the csv file for train data
test_data = pd.read_csv("Randombook_csv_test.csv", header=0,delimiter=",",quotechar='"', quoting=csv.QUOTE_ALL,names=['tweet','topic'])   # Read the csv file for train data

train_data_string=[]   # Initializing the string array to store appended train data
test_data_string=[]    # Initializing the string array to store appended test data

for i in xrange( 0, train_data['tweet'].size ):
    train_data_string.append(train_data['tweet'][i])    # Appending train data

for i in xrange( 0, test_data['tweet'].size ):
    test_data_string.append(test_data['tweet'][i])      # Appending test data


print "Creating Bag of Words representation"

Vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = 'english', max_features = 5000)

train_data_string_transformed=Vectorizer.fit_transform(train_data_string)   # fit_transform() does two functions: First, it fits the model and learns the vocabulary
                                                                            # Second, it transforms our training data into feature vectors

test_data_string_transformed=Vectorizer.transform(test_data_string)         # Only transforms our test data into feature vectors


tweets_train = train_data_string_transformed.toarray()                      # Convert the result to an array
tweets_test = test_data_string_transformed.toarray()


topics_train = numpy.array(train_data["topic"]).astype(int)                 # The topics of train data are stored in an array
topics_test = numpy.array(test_data["topic"]).astype(int)                   # The topics of test data are stored in an array

number_of_classes=6                                                         # Number of topics (class) are stored in a variable

topics_onehot_train = np_utils.to_categorical(topics_train, number_of_classes)   # Converts a class vector to binary class matrix.
topics_onehot_test = np_utils.to_categorical(topics_test, number_of_classes)


print('tweet_train shape:', tweets_train.shape)
print('tweet_test shape:', tweets_test.shape)
print('tweet_train shape:', topics_onehot_train.shape)
print('tweet_test shape:', topics_onehot_test.shape)

cvscores = []

print 'Bulding the model'

input_dimention = tweets_train.shape[1]
print input_dimention

model = Sequential()
model.add(Dense(256, input_dim=input_dimention,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(number_of_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



print("Training is going on")
model.fit(tweets_train, topics_onehot_train, epochs=10, batch_size=32,validation_data=(tweets_test,topics_onehot_test))


print("Generating test Predictions, Accurecy")
predited_topics = model.predict_classes(tweets_test, verbose=0)
accuracy_of_test = accuracy_score(test_data["topic"], predited_topics)
plot_model(model, to_file='model.png')


print  predited_topics

print 'Accurecy is ',accuracy_of_test*100,'%'
