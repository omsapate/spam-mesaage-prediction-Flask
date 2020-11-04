from flask import Flask,render_template,request,flash
import pandas as pd
from wtforms import Form
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

#key = os.urandom(12).hex()
app.config.from_object(__name__)
app.config['SECRET_KEY'] = "123456789"

# @app.route('/',methods=['POST'])
# def index():

# 	msg = request.form['message']
# 	result = ""
# 	if msg!="":
# 			model= joblib.load('pipeline.pkl')
# 			result = model.predict([msg])[0]
# 	return render_template("index.html",result=result)

def Model(msg):
	messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',names=['labels','message'])
	messages['labels'] = messages['labels'].map({'ham': 0, 'spam': 1})
	X= messages['message']
	y = messages['labels']
	cv = CountVectorizer()
	X = cv.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	classifier = MultinomialNB()
	classifier.fit(X_train,y_train)
	vector = cv.transform([msg]).toarray()

	pred = classifier.predict(vector)
	return pred


class ReusableForm(Form):
	#msg = TextAreaField("message")
	@app.route('/',methods=['GET','POST'])
	def index():

		form = ReusableForm(request.form)
		result = ""
		if request.method == 'POST':
			msg = request.form['message']
			#model= joblib.load('pipeline.pkl')
			#result = model.predict([msg])
			result = Model(msg)
			result = result[0]
			flash(" "+msg)

			# if result[0] == "spam":
			# 	flash(" "+msg)
			# 	#flash('Spam')
			# elif result[0] =='ham':
			# 	flash(" "+msg)
			# 	#flash('Not a Spam')

		return render_template("index.html",form=form, msg=result)

# @app.route('/predict', methods=['POST'])
# def predict():
# 	if request.method == 'POST':
# 		msg = request.form['message']
# 		#print(msg)
# 		result = ""
# 		if msg!="":
# 			model= joblib.load('pipeline.pkl')
# 			result = model.predict([msg])[0]

# 		#print(result)

# 		return render_template("index.html",result=result)


if __name__ == '__main__':
	app.run()