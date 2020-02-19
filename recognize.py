import cv2
import numpy as np
import os
import faceRecognition as fr
import openpyxl as op
import datetime as dt
from csv import reader , writer
import pandas as pd


#cap = cv2.VideoCapture(0)

camera = cv2.VideoCapture(0)
camera_height = 750
camera_width = 700
raw_frames = []


while(True):
    
    _, frame = camera.read()
    
    frame = cv2.flip(frame, 1)

    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height) 
    frame = cv2.resize(frame, (res, camera_height))

    
    cv2.rectangle(frame, (3, 3), (800, 650), (0, 255, 0), 2)

    
    cv2.imshow("Capturing frames", frame)

    key = cv2.waitKey(1)

    
    if key & 0xFF == ord("q"):
        # save the frame
        raw_frames.append(frame)
        print('1 key pressed - saved TYPE_1 frame')
        break
    #elif key & 0xFF == ord("1"):
        
        
camera.release()
cv2.destroyAllWindows()

for i, frame in enumerate(raw_frames):
    
    roi = frame[3+2:800-2, 3+2:650-2]
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    roi = cv2.resize(roi, (224, 224))

    path = 'images/{}.png'.format(i)

    cv2.imwrite(path , cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))


label1 = []
confidence1 = []
count = 0


test_img = cv2.imread("/home/avisek/Desktop/FaceRecognition/images/0.png")

faces_detected,gray_img = fr.faceDetection(test_img)
print(f"faces_detected : {faces_detected}")



# faces,face_id = fr.labels_for_training_data("/home/avisek/Desktop/FaceRecognition/dataset/training/Batch-2018-2019")
# face_recognizer = fr.train_classifier(faces,face_id)
# face_recognizer.save("trainedModel/Batch-2018-2019.yml")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("/home/avisek/Desktop/FaceRecognition/trainedModel/Batch-2018-2019.yml")

# face_recognizer = fr.train_classifier(faces,face_id)
# face_recognizer.save("trainedModel/Batch-2018-2019.yml")
# faces,face_id = fr.labels_for_training_data("/home/avisek/Desktop/FaceRecognition/dataset/training/Batch-2018-2019")

name = {35:"Atiab kalam",
        51:"Avisek shaw",
        60:"Agnibesh Mukherjee",
        64:"Abhishek Charan",
        37:"Madhurima Maji",
        32:"Arnab kumar Pati"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print(f"confidence : {confidence}")
    print(f"label : {label}")
    if confidence>150:
        fr.draw_rect(test_img,face)
        fr.put_text(test_img,"Not Registered",x,y)
    else:

        label1.append(int(label))
        #confidence1.append(confidence)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        fr.put_text(test_img,predicted_name,x,y)


resized_imag = cv2.resize(test_img,(1000,700))
cv2.imshow("face detection",resized_imag)
cv2.waitKey(0)
cv2.destroyAllWindows




print(f"label : {label1}")



    
df = pd.read_excel("attendanceSheet/cse-2018-2019-maths.xlsx",index=False)
df.to_csv('./current.csv')      



with open('current.csv', 'r') as read_obj, \
        open('cse-2018-2019-maths.csv', 'w', newline='') as write_obj:
    
    csv_reader = reader(read_obj)
    
    csv_writer = writer(write_obj)
    

    Date = dt.datetime.now()
    todaysDate = f"{Date.day}/{Date.month}/{Date.year}"
    for row in csv_reader:
        
        if (count==0):
            row.append(todaysDate)
        else:
            if (int(row[2]) not in label1):
                print("absent")
                status = "A"
            else:
                status = "P"
                print("present")
            row.append(status)
        # Add the updated row / list to the output file
        count = count + 1
        print(f"row[2] : {row[2]}")
        csv_writer.writerow(row)


read_file = pd.read_csv (r'cse-2018-2019-maths.csv')
read_file.to_excel (r'attendanceSheet/cse-2018-2019-maths.xlsx', index = None, header=True)
