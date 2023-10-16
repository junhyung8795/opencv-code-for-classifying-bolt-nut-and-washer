import cv2 
import numpy as np


cap = cv2.VideoCapture(0)

def mouse_callback(event, x, y, flags, param): 
    print("마우스 이벤트 발생, x:", x ," y:", y) # 이벤트 발생한 마우스 위치 출력
while True:
    ret,  frame = cap.read()
    cv2.setMouseCallback("Show", mouse_callback)
    if ret == False:
        print("촬영실패")
        break;
    # belt_location =     
    cv2.imshow("Show", frame)
    
    key = cv2.waitKey(1)
    if key== 27:
        break;
cap.release()
cv2.destroyAllWindows()