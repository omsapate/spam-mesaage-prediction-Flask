import nltk
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib


def Model():
	messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',names=['labels','message'])
	messages['length'] = messages['message'].apply(len)
		#bow_transfer = CountVectorizer(analyzer=text_process).fit(messages['message'])
		#message_bow = bow_transfer.transform(messages['message'])
	msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['labels'], test_size=0.3)
	pipeline = Pipeline([
	('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
	('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
	])

	pipeline.fit(msg_train,label_train)
		# prediction = pipeline.predict(msg_test)
		# print(classification_report(label_test,prediction))
	joblib.dump(pipeline,open("pipeline.pkl",'wb'))

# def Model_load(msg):
# 	pipeline = joblib.load("pipeline.pkl")
# 	result = pipeline.predict(msg)
# 	return result



def text_process(mess):
		nopunc = [char for char in mess if char not in string.punctuation]
		nopunc = ''.join(nopunc)
		return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

Model()
