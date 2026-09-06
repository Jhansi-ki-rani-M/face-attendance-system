import cv2
import os
import csv
from datetime import datetime


# ============================================================
# FILES / MODELS
# ============================================================

detector_model = "face_detection_yunet_2023mar.onnx"
recognizer_model = "face_recognition_sface_2021dec.onnx"

dataset_path = os.path.join("dataset", "Jhansi")

attendance_folder = "attendance"
attendance_file = os.path.join(
    attendance_folder,
    "attendance.csv"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(detector_model):
    print("ERROR: YuNet model not found!")
    print(f"Missing: {detector_model}")
    exit()

if not os.path.exists(recognizer_model):
    print("ERROR: SFace model not found!")
    print(f"Missing: {recognizer_model}")
    exit()

if not os.path.exists(dataset_path):
    print("ERROR: Jhansi dataset not found!")
    print(f"Missing: {dataset_path}")
    exit()


# ============================================================
# CREATE ATTENDANCE FOLDER
# ============================================================

os.makedirs(attendance_folder, exist_ok=True)


# ============================================================
# CREATE ATTENDANCE CSV
# ============================================================

if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time",
            "Status"
        ])


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

detector = cv2.FaceDetectorYN.create(
    detector_model,
    "",
    (320, 240),
    0.9,
    0.3,
    5000
)


# ============================================================
# LOAD FACE RECOGNIZER
# ============================================================

recognizer = cv2.FaceRecognizerSF.create(
    recognizer_model,
    ""
)


# ============================================================
# LOAD JHANSI'S FACE FEATURES
# ============================================================

known_features = []

print()
print("Loading Jhansi's registered faces...")

for filename in os.listdir(dataset_path):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_path = os.path.join(
            dataset_path,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        # Detect face in registered image
        _, faces = detector.detect(image)

        if faces is not None and len(faces) > 0:

            face = faces[0]

            try:

                # Align face
                aligned_face = recognizer.alignCrop(
                    image,
                    face
                )

                # Extract face feature
                feature = recognizer.feature(
                    aligned_face
                )

                known_features.append(
                    feature
                )

            except Exception:
                continue


print(
    f"Loaded {len(known_features)} "
    f"face samples for Jhansi."
)


if len(known_features) == 0:

    print()
    print("ERROR: No usable face samples found.")
    print("Run register_face.py again.")
    exit()


# ============================================================
# KEEP TRACK OF TODAY'S ATTENDANCE
# ============================================================

marked_today = set()

today = datetime.now().strftime(
    "%Y-%m-%d"
)

# Read existing attendance so we don't mark
# Jhansi again if she already attended today.

with open(
    attendance_file,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if (
            row.get("Name") == "Jhansi"
            and row.get("Date") == today
        ):

            marked_today.add(
                f"Jhansi_{today}"
            )


# ============================================================
# OPEN WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print()
print("========================================")
print("       FACE ATTENDANCE SYSTEM")
print("========================================")
print("Camera started.")
print("Look at the camera.")
print("Press Q to quit.")
print("========================================")
print()


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera.")
        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    _, faces = detector.detect(frame)


    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)


            # ------------------------------------------------
            # ALIGN FACE
            # ------------------------------------------------

            try:

                aligned_face = recognizer.alignCrop(
                    frame,
                    face
                )

                feature = recognizer.feature(
                    aligned_face
                )

            except Exception:

                continue


            # ------------------------------------------------
            # COMPARE WITH REGISTERED FACES
            # ------------------------------------------------

            best_score = 0.0

            for known_feature in known_features:

                score = recognizer.match(
                    known_feature,
                    feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                best_score = max(
                    best_score,
                    score
                )


            # ------------------------------------------------
            # RECOGNITION
            # ------------------------------------------------

            if best_score >= 0.363:

                name = "Jhansi"

                label = (
                    f"{name} "
                    f"({best_score:.2f})"
                )

                box_color = (
                    0,
                    255,
                    0
                )


                # --------------------------------------------
                # MARK ATTENDANCE
                # --------------------------------------------

                attendance_key = (
                    f"{name}_{today}"
                )


                if attendance_key not in marked_today:

                    current_time = datetime.now().strftime(
                        "%H:%M:%S"
                    )


                    with open(
                        attendance_file,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            name,
                            today,
                            current_time,
                            "Present"
                        ])


                    marked_today.add(
                        attendance_key
                    )


                    print(
                        f"Attendance marked: "
                        f"{name} at "
                        f"{current_time}"
                    )


            else:

                name = "Unknown"

                label = (
                    f"{name} "
                    f"({best_score:.2f})"
                )

                box_color = (
                    0,
                    0,
                    255
                )


            # ------------------------------------------------
            # DRAW FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )


            # ------------------------------------------------
            # DISPLAY NAME
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(y - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Face Recognition - Attendance",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEAN UP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Camera closed.")
print("Attendance file:")
print(attendance_file)