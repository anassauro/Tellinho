import cv2
import mediapipe as mp
from math import sqrt

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def distance(x1, y1, x0, y0):
  return sqrt((x1-x0)**2 + (y1-y0)**2)

piscando = False

def pisca(v1, d1, v2, d2, img):
    global piscando

    olhoDireito = v1 / d1
    olhoEsquerdo = v2 / d2

    if olhoDireito < 0.08 or olhoEsquerdo < 0.08:
        # Piscando os dois olhos
        if olhoDireito < 0.08 and olhoEsquerdo < 0.08 and piscando == False:
            piscando = True
            print('PISCOU')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
        if olhoDireito >= 0.12 and olhoEsquerdo >= 0.12 and piscando == True:
            piscando = False
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Piscando o olho direito
        if olhoDireito < 0.08 and olhoEsquerdo >= 0.12 and piscando == False:
            piscando = True
            print('Piscou Direito ;)')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
        elif olhoDireito >= 0.12 and olhoEsquerdo >= 0.12 and piscando == True:
            piscando = False

        # Piscando o olho esquerdo
        if olhoEsquerdo < 0.08 and olhoDireito >= 0.12 and piscando == False:
            piscando = True
            print('Piscou Esquerdo ;)')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
        elif olhoEsquerdo >= 0.12 and olhoDireito >= 0.12 and piscando == True:
            piscando = False
    else:
        piscando = False
        # print('Nao piscou :(')

    return img

while (cv2.waitKey(1) != ord(" ")):
  try:
    ret, frame = cap.read()
    cv2.imshow('aba1', frame)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    eye_dots = [130, 145, 159, 243, 359, 374, 386, 463]
    with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:
  # for idx, file in enumerate(IMAGE_FILES):
      image = frame
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print and draw face mesh landmarks on the image.
      if not results.multi_face_landmarks:
        continue
      annotated_image = image.copy()
      for face_landmarks in results.multi_face_landmarks:
        i=0
        positionsx = {}
        positionsy = {}
        for landmarks in face_landmarks.landmark:
          #marcando apenas os pontos de landmark dos olhos
          if i in eye_dots:
            shape = annotated_image.shape
            x = landmarks.x
            y = landmarks.y
            relative_x = int(x*shape[1])
            relative_y = int(y*shape[0])
            positionsx[str(i)] = relative_x
            positionsy[str(i)] = relative_y
            cv2.circle(annotated_image, (relative_x, relative_y), radius=1, color=(255,0,100), thickness=1)
          i+=1

#calculando as distancias dos pontos (1-esquerda, 2-direita)
        distv1 = distance(positionsx["159"], positionsy["159"], positionsx["145"], positionsy["145"])
        disth1 = distance(positionsx["130"], positionsy["130"], positionsx["243"], positionsy["243"])
        distv2 = distance(positionsx["386"], positionsy["386"], positionsx["374"], positionsy["374"])
        disth2 = distance(positionsx["463"], positionsy["463"], positionsx["359"], positionsy["359"])
        # pisca(distv1, disth1, distv2, disth2, annotated_image)
        # cv2.imshow('Aba', annotated_image)
        cv2.imshow('Aba', pisca(distv1, disth1, distv2, disth2, annotated_image))
        if cv2.waitKey(1) == ord('q'):
          break

  except KeyboardInterrupt:
    break

#tellinho.land()
