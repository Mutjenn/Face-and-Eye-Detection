import cv2

# Haarcascade sınıflayıcıları yüz ve göz tanıma için yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Kamera bağlantısını başlat
video_capture = cv2.VideoCapture(0)  # 0, varsayılan kamerayı temsil eder. Birden fazla kamera varsa 1, 2, vb. olarak değiştirilebilir.

while True:
    # Kameradan bir kare oku
    ret, frame = video_capture.read()

    # Görüntüyü gri tona dönüştür (Yüz ve göz tanıma için daha iyi)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Algılanan yüzlerin etrafına dikdörtgen çiz ve gözleri algıla
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Gözleri algıla
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Pencerede görüntüyü göster
    cv2.imshow('Face and Eye Detection', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
video_capture.release()
cv2.destroyAllWindows()
