from flask import Flask,render_template,url_for,request,flash
from wtforms import Form, TextAreaField
import joblib
import test_app
import os

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

class ReusableForm(Form):
	#msg = TextAreaField("message")
	@app.route('/',methods=['GET','POST'])
	def index():

		form = ReusableForm(request.form)
		result = ""
		if request.method == 'POST':
			msg = request.form['message']
			model= joblib.load('pipeline.pkl')
			result = model.predict([msg])
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
	app.run(debug = True)