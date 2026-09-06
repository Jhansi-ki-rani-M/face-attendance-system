
# Face Attendance System

<p align="center">
  <b>A smart, web-based attendance management system using Face Detection and Face Recognition.</b>
</p>

<p align="center">
  Built with Python • OpenCV • Flask • HTML • CSS • JavaScript
</p>

---

## Project Overview

The **Face Attendance System** is a web-based application designed to simplify student attendance management using computer vision.

The system uses **face detection and face recognition** to identify registered students through a webcam. Students can be registered by capturing face samples, and during attendance, the system recognizes the student and displays the result.

Attendance is **not marked automatically**. It is recorded only after the user explicitly chooses to mark attendance.

---

## Features

- **Student Registration**
  - Register new students using a webcam.
  - Capture multiple face samples for better recognition.

- **Face Detection**
  - Detect faces from webcam frames using OpenCV's YuNet model.

- **Face Recognition**
  - Recognize registered students using the SFace face recognition model.

- **Manual Attendance Marking**
  - Recognition and attendance marking are separate.
  - Attendance is recorded only after explicit confirmation.

- **Attendance Records**
  - View recorded attendance through the web interface.

- **Modern Web Interface**
  - Clean and responsive dashboard.
  - Light, System, and Dark theme support.

- **Privacy Protection**
  - Face datasets and attendance data are kept locally.
  - Personal face images are excluded from the public GitHub repository.

---

# Application Screenshots

## Dashboard

<p align="center">
  <img width="945" height="414" alt="Screenshot 2026-09-06 215056" src="https://github.com/user-attachments/assets/ab2bbf92-0d3d-418b-a622-89c43863d62d" />
  
  <img width="947" height="410" alt="Screenshot 2026-09-06 215108" src="https://github.com/user-attachments/assets/ae6133f1-46d2-428e-b426-66304d0f4958" />

</p>

## Student Registration

<p align="center">
  <img width="943" height="412" alt="Screenshot 2026-09-06 215124" src="https://github.com/user-attachments/assets/d1090537-9f0a-41dc-9b77-3a8c698c5db7" />
</p>

## Face Recognition & Attendance

<p align="center">
  <img width="941" height="409" alt="Screenshot 2026-09-06 215202" src="https://github.com/user-attachments/assets/db90be18-2c24-4ca9-b68e-9d2040b93788" />
</p>

## Attendance Records

<p align="center">
  <img width="946" height="407" alt="Screenshot 2026-09-06 215215" src="https://github.com/user-attachments/assets/860ca3dc-2b2e-45db-aa4c-adb408f78111" />
</p>

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Flask | Web application framework |
| OpenCV | Computer vision and image processing |
| YuNet | Face detection |
| SFace | Face recognition |
| HTML | Web page structure |
| CSS | Web page styling |
| JavaScript | Webcam interaction and frontend functionality |

---

# How the System Works

The system follows a simple workflow:

```text
                ┌─────────────────────┐
                │     Open Web App    │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      Register Student          Take & Mark Attendance
              │                         │
              ▼                         ▼
     Capture Face Samples       Capture Webcam Frame
              │                         │
              ▼                         ▼
      Store Face Samples          Detect Face using
              │                       YuNet
              │                         │
              │                         ▼
              │                  Recognize Face
              │                    using SFace
              │                         │
              │                         ▼
              │                  Display Student
              │                   & Confidence
              │                         │
              │                         ▼
              │                  User Explicitly
              │                  Marks Attendance
              │                         │
              └────────────┬────────────┘
                           ▼
                  Attendance Records
```

---

# Project Structure

```text
face-attendance-system/
│
├── app.py
├── capture_faces.py
├── face_recognition.py
├── recognize.py
├── register_face.py
├── requirements.txt
│
├── face_detection_yunet_2023mar.onnx
├── face_recognition_sface_2021dec.onnx
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   └── records.html
│
├── screenshots/
│   ├── dashboard.png
│   ├── register.png
│   ├── attendance.png
│   └── records.png
│
├── .gitignore
├── .gitattributes
└── README.md
```

> **Note:** The `dataset/`, `venv/`, and attendance data are intentionally excluded from the public repository using `.gitignore`.

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Jhansi-ki-rani-M/face-attendance-system.git
cd face-attendance-system
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

## 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Application

```bash
python app.py
```

The application will start on the local Flask server.

---

# How to Use

## Register a Student

1. Open the application in your browser.
2. Select **Register Student**.
3. Enter the student's name.
4. Allow webcam access.
5. The system captures multiple face samples.
6. The face samples are stored locally for recognition.

## Take & Mark Attendance

1. Select **Take & Mark Attendance**.
2. Allow webcam access.
3. The system detects the face using YuNet.
4. The system recognizes the registered student using SFace.
5. The student's name and confidence score are displayed.
6. Click the attendance button to explicitly mark attendance.
7. The attendance record is stored in the attendance CSV file.

## View Attendance Records

1. Select **Attendance Records**.
2. The system displays the recorded attendance for the current day.

---

# Privacy & Security

- Student face datasets are stored locally.
- Face images are not included in the public GitHub repository.
- Attendance data is excluded from the public repository.
- The `dataset/` and attendance data are protected using `.gitignore`.
- Attendance is marked only after explicit user action.
- No student face data is automatically uploaded to external services.

---

# Project Objectives

The main objectives of this project are:

- To automate the student identification process using face recognition.
- To simplify attendance management through a web-based interface.
- To provide a reliable method for recognizing registered students.
- To separate face recognition from attendance marking to prevent accidental attendance records.
- To maintain attendance information in a simple and accessible format.
- To demonstrate the practical application of computer vision and machine learning.

---

# Future Improvements

- Add a database for storing student and attendance information.
- Improve face recognition accuracy under different lighting conditions.
- Add administrator authentication.
- Add monthly and yearly attendance reports.
- Export attendance records to Excel or PDF.
- Deploy the system online.
- Add support for multiple cameras.
- Add improved student management features.

---

# Learning Outcomes

Through this project, I gained practical experience in:

- Python programming
- Computer vision
- Face detection and face recognition
- OpenCV
- Flask web development
- HTML and CSS
- JavaScript
- CSV-based data management
- Machine learning models
- Git and GitHub
- Building and integrating a complete software application

---

# Author

**Jhansi M**

B.Tech – Computer Science Engineering  
Artificial Intelligence & Machine Learning

GitHub:  
https://github.com/Jhansi-ki-rani-M
