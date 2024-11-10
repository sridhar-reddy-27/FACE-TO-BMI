from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import numpy as np
import cv2
import os
from Model import get_model

main = tkinter.Tk()
main.title("FACE TO BMI") #designing main screen
main.geometry("800x700")

global filename
global image
global model
global faceCascade

def loadModel():
    global model
    global faceCascade
    textarea.delete('1.0', END)
    cascPath = "model/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    model = get_model(ignore_age_weights=True)
    model.load_weights('model/bmi_model_weights.h5')
    textarea.insert(END,"BMI Prediction & Face Detection Models loaded\n")
    


def upload(): 
    global filename
    textarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="images")
    textarea.insert(END,filename+" image loaded")
    
    
def predictBMI():
    global model
    global faceCascade
    global filename
    textarea.delete('1.0', END)
    frame = cv2.imread(filename)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    img = None
    for (x, y, w, h) in faces:
        img = frame[y:y + (h+10), x:x + (w+10)]
    if img is not None:
        img = cv2.resize(img,(224,224))
        temp = []
        temp.append(img)
        temp = np.asarray(temp)
        prediction = model.predict(temp)
        bmi = prediction[0][0]
        bmi = bmi / 20
        result = ''
        if bmi > 15 and bmi < 25:
            textarea.insert(END,"Your BMI predicted as : "+str(bmi)+"\n")
            textarea.insert(END,"Suitable Quoted Policy Based on Predicted BMI is 25 Lakhs\n")
            result = "Suitable Quoted Policy Based on Predicted BMI is 25 Lakhs"
        if bmi >= 25 and bmi < 30:
            textarea.insert(END,"Your BMI predicted as : "+str(bmi)+"\n")
            textarea.insert(END,"Suitable Quoted Policy Based on Predicted BMI is 20 Lakhs\n")
            result = "Suitable Quoted Policy Based on Predicted BMI is 20 Lakhs"
        if bmi >= 30 and bmi < 40:
            textarea.insert(END,"Your BMI predicted as : "+str(bmi)+"\n")
            textarea.insert(END,"Suitable Quoted Policy Based on Predicted BMI is 15 Lakhs\n")
            result = "Suitable Quoted Policy Based on Predicted BMI is 15 Lakhs"
        if bmi >= 40:
            textarea.insert(END,"Your BMI predicted as : "+str(bmi)+"\n")
            textarea.insert(END,"Suitable Quoted Policy Based on Predicted BMI is 10 Lakhs\n")
            result = "Suitable Quoted Policy Based on Predicted BMI is 10 Lakhs"
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,400))
        cv2.putText(img, 'Predicted BMI based on facial features is : '+str(bmi), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.putText(img, result, (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Predicted BMI based on facial features is : '+str(bmi), img)
        cv2.waitKey(0)    
    else:
        textarea.insert(END,"Facial Features not detected in uploaded image\n")        
            



def exit():
    global main
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='FACE TO BMI', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')
model = Button(main, text="Generate & Load BMI & Face Detection Models", command=loadModel)
model.place(x=200,y=100)
model.config(font=font1)  

uploadimage = Button(main, text="Upload Image", command=upload)
uploadimage.place(x=200,y=150)
uploadimage.config(font=font1) 

bmiimage = Button(main, text="Run Face & BMI Detection Algorithm", command=predictBMI)
bmiimage.place(x=200,y=200)
bmiimage.config(font=font1)

exitapp = Button(main, text="Exit", command=exit)
exitapp.place(x=200,y=250)
exitapp.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=15,width=60)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=300)
textarea.config(font=font1)

main.config(bg='light coral')
main.mainloop()
