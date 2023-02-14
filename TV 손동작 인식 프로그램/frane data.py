import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

number = '133'
# 4(펼치기), 20(왼쪽에서 오른쪽),49(오른쪽에서 왼쪽),33(위로),133(아래로)
cap = cv2.VideoCapture('train/TRAIN_' + number + '.mp4')

# 파일 생성
os.mkdir('trainframe/'+str(number))

# 영상 파일저장
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
out = cv2.VideoWriter('trainframe/'+number+'/video.avi', fourcc, fps, (w, h))

#파일 생성

# base_img = cv2.imread('base_img.jpg')
num =0
while True:
    ret,frame = cap.read()
    if not ret :
        break
    base_img = cv2.imread('base_img.png')
    img = cv2.resize(img, (360, 360))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (360, 360))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(base_img,
                                      hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(img,
    #         hand_landmarks, mp_hands.HAND_CONNECTIONS)

    out.write(base_img)

    if ret:
        cv2.imshow('frame', base_img)
        #이미지의 각 이름을 자동으로 지정
        path = 'trainframe/'+str(number)+'/snapshot_' + str(num) + '.jpg'
        cv2.imwrite(path, base_img) #영상 -> 이미지로 저장
        if cv2.waitKey(1) == ord('q'):
            break
    num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
