import cv2 
import numpy as np

import torch
original_model = torch.load('./boltNutScrew.pt', map_location=torch.device('cpu'))
scripted_model = torch.jit.script(original_model)
scripted_model.save('./boltNutScrew_scripted.pt')

net = cv2.dnn.readNetFromTorch('./boltNutScrew_scripted.pt')

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
    MEAN_VALUE = [103.939, 116.779, 123.680]
    blob = cv2.dnn.blobFromImage(belt_location, mean=MEAN_VALUE)
    net.setInput(blob)
    output = net.forward()
    output = output.squeeze().transpose((1, 2, 0))
    output += MEAN_VALUE
    
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

    cv2.imshow('result', output)
    
    gray_belt = cv2.cvtColor(belt_location, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_belt, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(belt_location, [box], 0, (0, 255, 0), 3)
        # if area > 500:
        #     cv2.rectangle(belt_location, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # elif 100 < area < 500:
        #     cv2.rectangle(belt_location, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(belt_location, str(area), (x, y), 1, 1, (0, 255, 0))

    cv2.imshow("kk", belt_location)
    
    
    key = cv2.waitKey(1)
    if key== 27:
        break;
cap.release()
cv2.destroyAllWindows()