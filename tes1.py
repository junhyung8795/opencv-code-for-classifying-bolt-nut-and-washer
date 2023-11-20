import cv2 
import numpy as np





cap = cv2.VideoCapture(0)

def mouse_callback(event, x, y, flags, param): 
    print("마우스 이벤트 발생, x:", x ," y:", y) # 이벤트 발생한 마우스 위치 출력
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

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

    
    
    belt_location = frame
    
    blur = cv2.GaussianBlur(belt_location, (5,5), 0)
    img_mask = fgbg.apply(blur, learningRate=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
    
    
    gray_belt = cv2.cvtColor(belt_location, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img_mask, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_mask, [box], 0, (0, 255, 0), 3)
        if area > 500:
            cv2.rectangle(img_mask, (x, y), (x + w, y + h), (255, 255, 255), 3)
        elif 100 < area < 500:
            cv2.rectangle(img_mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(img_mask, str(area), (x, y), 1, 1, (0, 255, 0))

    cv2.imshow("kk", belt_location)
    cv2.imshow("kkkk", img_mask)
    
    key = cv2.waitKey(1)
    if key== 27:
        break;
cap.release()
cv2.destroyAllWindows()