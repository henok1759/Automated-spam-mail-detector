from flask import Flask,flash, request, redirect, url_for, render_template,session
import pickle
import pytesseract
import io
from PIL import Image
import speech_recognition as sr


# load the model from disk
filename = 'nlp model.pkl'
mb = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))


    
    
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
 


@app.route('/')
@app.route('/home')
def home():
	return render_template('index.html')
@app.route('/text')
def text():
	return render_template('text.html')


@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = mb.predict(vect)
	return render_template('result.html',prediction = my_prediction)


@app.route('/extract', methods=['POST'])
def extract():
    if request.method == 'POST':
        
        image_data = request.files['file'].read()
        scanned_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_data)))
        data = [scanned_text]
        vect = cv.transform(data).toarray()
        my_prediction = mb.predict(vect)
    return render_template('result.html',prediction = my_prediction)

@app.route('/voice')
def voice():
    transcript = ""
    if request.method == "POST":
        file = request.files["files"]
        
        recognizer = sr.Recognizer()
        audioFile = sr.AudioFile(file)
        with audioFile as source:
             data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        vect = cv.transform(transcript).toarray()
        my_prediction = mb.predict(vect)
    return render_template('result.html',prediction = my_prediction)
	

       
       
         

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/introduction')
def introduction():
	return render_template('introduction.html')
@app.route('/backend')
def backend():
	return render_template('backend.html')
@app.route('/frontend')
def frontend():
	return render_template('frontend.html')
@app.route('/framework')
def framework():
	return render_template('framework.html')
@app.route('/reference')
def reference():
	return render_template('reference.html')
@app.route('/picture')
def picture():
	return render_template('pic.html')


if __name__ == '__main__':
    # Setup Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    app.run(debug=True)