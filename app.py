import re
import numpy as np
import os
from flask import Flask, app,request,render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
#Loading the model
model=load_model(r"Crime_classification.h5")

app=Flask(__name__)


#default home page or route
@app.route('/')
def Main_home():
    return render_template('Main_home.html')

@app.route('/Prediction.html')
def Prediction():
    return render_template('Prediction.html')

@app.route('/Home_des.html')
def home():
    return render_template("Home_des.html")

@app.route('/a',methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__) #getting the current path i.e where app.py is present
        #print("current path",basepath)
        filepath=os.path.join(basepath,'upload',f.filename) #from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        #print("upload folder is",filepath)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(64,64,3))
        x=image.img_to_array(img)#img to array
        x=np.expand_dims(x,axis=0)#used for adding one more dimension
        #print(x)
        pred=np.argmax(model.predict(x), axis =1) #instead of predict_classes(x) we can use predict(X) ---->predict_classes(x) gave error
        #print("prediction is ",prediction)
        index=['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','Normal','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']
        #result=str(index[prediction])
        a = index[int(pred)]
        
        return render_template('Prediction.html',pred=a)
        

""" Running our application """
if __name__ == "__main__":
    app.run(debug=True,port=8000)