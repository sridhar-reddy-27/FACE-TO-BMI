from Model import get_model
import cv2
import numpy as np

def get_trained_model():
    weights_file = 'bmi_model_weights.h5'
    model = get_model(ignore_age_weights=True)
    model.load_weights(weights_file)
    return model

model = get_trained_model()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

for i in range(1,9):
    frame = cv2.imread('images/'+str(i)+'.png')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    img = None
    for (x, y, w, h) in faces:
        img = frame[y:y + (h+10), x:x + (w+10)]
    img = cv2.resize(img,(224,224))
    temp = []
    temp.append(img)
    temp = np.asarray(temp)
    prediction = model.predict(temp)
    print(prediction)
    #print(str(prediction[0][0]))
