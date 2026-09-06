import cv2

# Path to the YuNet face detection model
model_path = "face_detection_yunet_2023mar.onnx"

# Open webcam
camera = cv2.VideoCapture(0)

# Get webcam resolution
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create YuNet face detector
face_detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (width, height),
    0.9,
    0.3,
    5000
)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the camera.")
        break

    # Tell the detector the current frame size
    face_detector.setInputSize((frame.shape[1], frame.shape[0]))

    # Detect faces
    _, faces = face_detector.detect(frame)

    # Draw results
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display label
            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Attendance System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()