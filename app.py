import cv2
import numpy as np

# ---------------- LOAD CASCADES ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

if face_cascade.empty() or smile_cascade.empty():
    print("Error loading Haar Cascade files")
    exit()

# ---------------- OPEN CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

# ---------------- HSV COLOR RANGES ----------------
blue_lower = np.array([100, 150, 0])
blue_upper = np.array([140, 255, 255])

red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -------- BLUE DETECTION --------
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in blue_contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Blue Detected",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # -------- RED DETECTION (BLINK) --------
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blink = int(cv2.getTickCount() / cv2.getTickFrequency()) % 2 == 0

    for cnt in red_contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            if blink:
                cv2.line(frame,
                         (x + w//2, y),
                         (x + w//2, y + h),
                         (0, 0, 255), 3)
            cv2.putText(frame, "Red Detected (Blink)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # -------- FACE & MOOD DETECTION --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=25, minSize=(25,25)
        )

        mood = "Happy" if len(smiles) > 0 else "Serious"
        mind = "Open / Positive" if mood == "Happy" else "Focused / Serious"

        cv2.putText(frame, f"Mood: {mood}",
                    (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Mind: {mind}",
                    (x, y+h+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # -------- DISPLAY --------
    cv2.imshow("AI Color, Face & Mood Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()