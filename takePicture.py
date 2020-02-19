"""
copyright Avisek shaw
"""
import numpy as np
import cv2


camera = cv2.VideoCapture(0)
camera_height = 600
raw_frames = []


while(True):
    
    _, frame = camera.read()
    
    frame = cv2.flip(frame, 1)

    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height) 
    frame = cv2.resize(frame, (res, camera_height))

    
    cv2.rectangle(frame, (51, 37), (725, 555), (0, 255, 0), 2)

    
    cv2.imshow("Capturing frames", frame)

    key = cv2.waitKey(1)

    
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("1"):
        # save the frame
        raw_frames.append(frame)
        print('1 key pressed - saved TYPE_1 frame')
        
camera.release()
cv2.destroyAllWindows()

for i, frame in enumerate(raw_frames):
    
    roi = frame[37+2:555-2, 51+2:650-2]
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    roi = cv2.resize(roi, (224, 224))

    cv2.imwrite('images/{}.png'.format(i), cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))