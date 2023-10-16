import cv2 
import numpy as np


cap = cv2.VideoCapture(0)

def mouse_callback(event, x, y, flags, param): 
    print("마우스 이벤트 발생, x:", x ," y:", y) # 이벤트 발생한 마우스 위치 출력
while True:
    ret,  frame = cap.read()
    if ret == False:
        print("촬영실패")
        break;
    # belt_location =     
    # 왼쪽 위 꼭짓점 x = 452 , y = 233 
    # 오른쪽 아래 꼭짓점 x = 893, y = 463
    
    # cv2.imshow("Show", frame)
    # cv2.imshow("belt", belt_location)
    belt_location = frame[233: 463, 452: 893]
    gray_belt = cv2.cvtColor(belt_location, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_belt, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(gray_belt, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("kk", belt_location)

    cv2.imshow("Gray_Belt", gray_belt)
    cv2.imshow("TTT", threshold)
    

    
    key = cv2.waitKey(1)
    if key== 27:
        break;
cap.release()
cv2.destroyAllWindows()