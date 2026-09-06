import cv2
import os

# Ask for person's name
name = input("Enter person's name: ").strip()

# Folder where this person's face images will be saved
save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

# YuNet model
model_path = "face_detection_yunet_2023mar.onnx"

# Open webcam
camera = cv2.VideoCapture(0)

# Create face detector
face_detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 240),
    0.9,
    0.3,
    5000
)

count = 0
max_images = 30

print()
print(f"Registering face for: {name}")
print("Look at the camera.")
print("Press Q to stop.")

while count < max_images:

    success, frame = camera.read()

    if not success:
        print("Could not access the camera.")
        break

    height, width = frame.shape[:2]

    # Update detector size
    face_detector.setInputSize((width, height))

    # Detect face
    _, faces = face_detector.detect(frame)

    if faces is not None and len(faces) > 0:

        # Use first detected face
        face = faces[0]

        x, y, w, h = face[:4].astype(int)

        # Keep coordinates inside image
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Crop face
        face_image = frame[y:y+h, x:x+w]

        if face_image.size > 0:

            filename = os.path.join(
                save_path,
                f"{name}_{count + 1}.jpg"
            )

            cv2.imwrite(filename, face_image)

            count += 1

            cv2.putText(
                frame,
                f"{name}: {count}/{max_images}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print()
print(f"Registration complete for {name}!")
print(f"{count} images saved in {save_path}")