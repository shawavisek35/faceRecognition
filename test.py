import cv2
import numpy as np
import os
import faceRecognition as fr
import openpyxl as op
import datetime as dt
from csv import reader , writer
import pandas as pd


cap = cv2.VideoCapture(0)


label1 = []
confidence1 = []
count = 0


#test_img = cv2.imread("/home/avisek/Desktop/FaceRecognition/images/groupPhoto.jpeg")





face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("/home/avisek/Desktop/FaceRecognition/trainedModel/Batch-2018-2019.yml")

name = {35:"Atiab kalam",
        51:"Avisek shaw",
        60:"Agnibesh Mukherjee",
        64:"Abhishek Charan",
        37:"Madhurima Maji",
        32:"Arnab kumar Pati"}




# resized_imag = cv2.resize(test_img,(1000,700))
# cv2.imshow("face detection",resized_imag)
# cv2.waitKey(0)
# cv2.destroyAllWindows




 


print(f"label : {label1}")


    
df = pd.read_excel("attendanceSheet/cse-2018-2019-maths.xlsx",index=False)
df.to_csv('./current.csv')      






while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('frame')

    faces_detected,gray_img = fr.faceDetection(frame)
    print(f"faces_detected : {faces_detected}")

    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h,x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)
        print(f"confidence : {confidence}")
        print(f"label : {label}")
        if confidence>100:
            continue
        label1.append(int(label))
        #confidence1.append(confidence)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        fr.put_text(test_img,predicted_name,x,y)

    

    # Display the resulting frame
    
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
