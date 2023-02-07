from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import cv2



app = Flask(__name__)
model = load_model('model21.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

           
# Function to load and prepare the image in right shape
def read_image(filename):

    frame = cv2.imread(filename)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray_frame,(299,299),interpolation=cv2.INTER_AREA) 
    roi = roi.astype('float')/255
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis = 0)
    return roi

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            global file_path
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction=model.predict(img)
            global classes_x
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
                Emotion = "Angry"
            elif classes_x == 1:
                Emotion = "Fear"
            elif classes_x == 2:
                Emotion = "Happy"
            else:
                Emotion = "Relaxed"
            return render_template('predict.html',  Emotion= Emotion,prob=class_prediction, user_image = file_path)
        else:
            return  render_template('error.html')

@app.route('/view',methods=['GET','POST'])
def view():
    if request.method == 'POST':
        if classes_x == 0:
            Emotion = "Angry"
            with open("static/texts/Angry.txt", 'r') as f:
                Tasks = f.read()
        elif classes_x == 1:
            Emotion = "Fear"
            with open("static/texts/Fear.txt", 'r') as f:
                  Tasks = f.read()
        elif classes_x == 2:
            Emotion = "Happy"
            with open("static/texts/Happy.txt", 'r') as f:
                Tasks = f.read()
        else:
            Emotion = "Relaxed"
            with open("static/texts/Relaxed.txt", 'r') as f:
                Tasks = f.read()
        return render_template('view.html',  Emotion= Emotion, user_image = file_path,Task= Tasks)
    else:
            return "Unable to read the file. Please check file extension"
       
      
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)