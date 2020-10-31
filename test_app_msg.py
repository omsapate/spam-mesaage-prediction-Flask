import test_app
import joblib

msg = "You have win a lottery! Please contact to claim reward."
pipeline = joblib.load("pipeline.pkl")
result = pipeline.predict([msg])



print(result[0])