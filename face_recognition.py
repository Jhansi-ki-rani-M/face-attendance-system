import cv2
import os


# ============================================================
# MODELS
# ============================================================

DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"

DATASET_PATH = "dataset"

# Recognition threshold
THRESHOLD = 0.363


# ============================================================
# LOAD MODELS
# ============================================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 240),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)


# ============================================================
# LOAD ALL REGISTERED STUDENTS
# ============================================================

def load_known_faces():

    known_faces = {}

    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found.")
        return known_faces

    # Go through every student folder
    for student_name in os.listdir(DATASET_PATH):

        student_folder = os.path.join(
            DATASET_PATH,
            student_name
        )

        # Skip files
        if not os.path.isdir(student_folder):
            continue

        print(f"Loading faces for {student_name}...")

        student_features = []

        # Go through student's images
        for filename in os.listdir(student_folder):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                student_folder,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            height, width = image.shape[:2]

            detector.setInputSize(
                (width, height)
            )

            # Detect face
            _, faces = detector.detect(image)

            if faces is None or len(faces) == 0:
                continue

            face = faces[0]

            try:

                # Align face
                aligned_face = recognizer.alignCrop(
                    image,
                    face
                )

                # Extract feature
                feature = recognizer.feature(
                    aligned_face
                )

                student_features.append(feature)

            except Exception as e:

                print(
                    f"Could not process {filename}"
                )

        if len(student_features) > 0:

            known_faces[student_name] = student_features

            print(
                f"  Loaded {len(student_features)} "
                f"samples for {student_name}"
            )

        else:

            print(
                f"  No usable face samples for "
                f"{student_name}"
            )

    print()
    print(
        f"Total registered students: "
        f"{len(known_faces)}"
    )

    return known_faces


# ============================================================
# RECOGNIZE A FACE
# ============================================================

def recognize_face(frame, face, known_faces):

    try:

        # Align detected face
        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        # Extract feature
        feature = recognizer.feature(
            aligned_face
        )

    except Exception:

        return "Unknown", 0.0


    best_name = "Unknown"
    best_score = 0.0


    # Compare against every student
    for student_name, features in known_faces.items():

        student_best_score = 0.0

        for known_feature in features:

            score = recognizer.match(
                known_feature,
                feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            student_best_score = max(
                student_best_score,
                score
            )

        # Check whether this student is the
        # best match so far
        if student_best_score > best_score:

            best_score = student_best_score
            best_name = student_name


    # Apply recognition threshold
    if best_score >= THRESHOLD:

        return best_name, best_score

    return "Unknown", best_score


# ============================================================
# TEST THE DATABASE
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("     FACE DATABASE")
    print("========================================")

    known_faces = load_known_faces()

    print()

    if len(known_faces) == 0:

        print("No registered students found.")

    else:

        print("Registered students:")

        for name in known_faces:

            print(
                f"  ✓ {name}: "
                f"{len(known_faces[name])} samples"
            )

    print()