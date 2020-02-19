import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier("harcascade/haarcascade_frontalface_default.xml")
    faces = face_haar_cascade.detectMultiScale(gray_img , scaleFactor=1.32 , minNeighbors=5)

    return faces,gray_img


def labels_for_training_data(directory):
    faces = []
    face_id = []

    for root,sub_directory,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping this file.....\n")
                continue
            
            id = os.path.basename(root)
            img_path = os.path.join(root,filename)
            print(f"img_path = {img_path}")
            print(f"id : {id}")
            test_img = cv2.imread(img_path)

            if test_img is None:
                print("Image not loaded properly ")
                continue

            faces_rect,gray_img = faceDetection(test_img)  
            if(len(faces_rect)!=1):
                continue

            (x,y,w,h) = faces_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            print(id)
            try:
                face_id.append(int(id))
            except:
                print("Invalid id")
                continue

    return faces , face_id


def train_classifier(faces,face_id):
    face_recoznizer = cv2.face.LBPHFaceRecognizer_create()
    face_recoznizer.train(faces,np.array(face_id))
    return face_recoznizer

def draw_rect(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX , 0.4,(255,0,0),2)


            