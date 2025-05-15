import cv2
import datetime
import os
import time

# === Setup Directories ===
output_dir = "captured_faces"
os.makedirs(output_dir, exist_ok=True)

# === Face Detection using Haar Cascade ===
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Video Writer Setup ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# === Face Capture Control ===
photo_counter = 1
last_capture_time = datetime.datetime.min
capture_interval = datetime.timedelta(seconds=5)

# === Frame Rate Limit ===
frame_delay = 0.03  # ~30 FPS

print("[INFO] Starting webcam... Press 'q' or ESC to quit.")

while True:
    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    current_time = datetime.datetime.now()

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if current_time - last_capture_time > capture_interval:
            face_crop = img[y:y + h, x:x + w]
            filename = f"{output_dir}/face_{photo_counter}_{i+1}.png"
            cv2.imwrite(filename, face_crop)
            print(f"[INFO] Saved face to: {filename}")
    
    if len(faces) > 0 and current_time - last_capture_time > capture_interval:
        photo_counter += 1
        last_capture_time = current_time

    # Timestamp overlay
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show and save frame
    cv2.imshow('Face Detection', img)
    out.write(img)

    key = cv2.waitKey(1)
    if key & 0xFF in [ord('q'), 27]:
        break

    time.sleep(frame_delay)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Finished.")
