from flask import Flask, render_template, render_template_string, request, jsonify
import cv2
import os
import base64
import numpy as np
import csv
from datetime import datetime

from face_recognition import load_known_faces, recognize_face


app = Flask(__name__)

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Attendance System</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0;min-height:100%;font-family:"Segoe UI",Arial,sans-serif}
html{background:#eefaf8}
body{background:var(--bg);color:var(--text)}
:root{--bg:#eefaf8;--panel:#fff;--panel2:#f3fbfa;--text:#173637;--muted:#647777;--teal:#087f7a;--teal2:#0b918b;--border:#d5e9e7;--shadow:0 18px 45px rgba(8,78,75,.12)}
:root[data-theme="light"]{--bg:#eefaf8;--panel:#fff;--panel2:#f3fbfa;--text:#173637;--muted:#647777;--teal:#087f7a;--teal2:#0b918b;--border:#d5e9e7;--shadow:0 18px 45px rgba(8,78,75,.12)}
:root[data-theme="dark"]{--bg:#071918;--panel:#102625;--panel2:#15302e;--text:#e9fffc;--muted:#a7c2c0;--teal:#5ee8dc;--teal2:#7af0e7;--border:#244844;--shadow:0 18px 45px rgba(0,0,0,.32)}
@media(prefers-color-scheme:dark){:root[data-theme="system"]{--bg:#071918;--panel:#102625;--panel2:#15302e;--text:#e9fffc;--muted:#a7c2c0;--teal:#5ee8dc;--teal2:#7af0e7;--border:#244844;--shadow:0 18px 45px rgba(0,0,0,.32)}}
.app-shell{min-height:100vh;display:flex;flex-direction:column;background:linear-gradient(135deg,#e8faf7 0%,#f8ffff 100%)}
:root[data-theme="dark"] .app-shell{background:linear-gradient(135deg,#071918,#0c2220)}
@media(prefers-color-scheme:dark){:root[data-theme="system"] .app-shell{background:linear-gradient(135deg,#071918,#0c2220)}}
.topbar{position:sticky;top:0;z-index:20;min-height:82px;padding:18px 34px;display:flex;align-items:center;justify-content:space-between;gap:20px;background:rgba(255,255,255,.88);border-bottom:1px solid var(--border);backdrop-filter:blur(14px)}
:root[data-theme="dark"] .topbar{background:rgba(8,31,30,.92)}
@media(prefers-color-scheme:dark){:root[data-theme="system"] .topbar{background:rgba(8,31,30,.92)}}
.brand{font-size:24px;font-weight:800;letter-spacing:-.4px;color:var(--text)}.tagline{margin-top:3px;font-size:14px;color:var(--muted)}
.menu-btn{width:46px;height:46px;border-radius:12px;border:1px solid var(--border);background:var(--panel);color:var(--teal);font-size:25px;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 16px rgba(0,0,0,.06)}
.menu-btn:hover{transform:translateY(-1px);background:var(--panel2)}
.main-content{width:min(980px,calc(100% - 40px));margin:0 auto;padding:58px 0 70px;flex:1}.hero{text-align:center;margin-bottom:34px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:2px;color:var(--teal);margin-bottom:12px}.hero h1{margin:0;color:var(--text);font-size:clamp(32px,5vw,52px);line-height:1.08;letter-spacing:-1.5px}.hero h1 span{color:var(--teal)}.hero p{max-width:720px;margin:18px auto 0;color:var(--muted);font-size:17px;line-height:1.6}
.dashboard-grid{display:flex;flex-direction:column;gap:20px}.dash-card{background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:28px 30px;box-shadow:var(--shadow);transition:transform .2s ease,box-shadow .2s ease}.dash-card:hover{transform:translateY(-3px);box-shadow:0 22px 50px rgba(8,78,75,.16)}
.card-icon{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:var(--panel2);color:var(--teal);font-size:25px;font-weight:700;margin-bottom:18px}.dash-card h2{margin:0;color:var(--text);font-size:25px}.dash-card p{margin:10px 0 22px;color:var(--muted);font-size:16px;line-height:1.5}
.primary-btn,.dashboard-btn,.back-btn,.go-dashboard,.dashboard-button{display:inline-flex!important;align-items:center;justify-content:center;gap:9px;padding:12px 20px!important;border-radius:11px!important;background:var(--teal)!important;color:#fff!important;border:0!important;outline:0!important;text-decoration:none!important;font-size:15px!important;font-weight:700!important;box-shadow:0 7px 18px rgba(8,127,122,.18)!important;cursor:pointer;transition:.2s ease}.primary-btn:hover,.dashboard-btn:hover,.back-btn:hover,.go-dashboard:hover,.dashboard-button:hover{background:var(--teal2)!important;transform:translateY(-1px)}.primary-btn span{font-size:18px}
footer{padding:18px 30px;text-align:center;color:var(--muted);font-size:13px;border-top:1px solid var(--border);background:rgba(255,255,255,.45)}:root[data-theme="dark"] footer{background:rgba(8,31,30,.45)}@media(prefers-color-scheme:dark){:root[data-theme="system"] footer{background:rgba(8,31,30,.45)}}
.sidebar{position:fixed;top:0;right:-330px;z-index:50;width:320px;height:100vh;padding:24px 20px;background:var(--panel);border-left:1px solid var(--border);box-shadow:-20px 0 45px rgba(0,0,0,.18);transition:right .25s ease}.sidebar.open{right:0}.sidebar-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px}.side-title{font-size:20px;font-weight:800;color:var(--text)}
.close-btn,.about-close{width:38px;height:38px;border:1px solid var(--border);border-radius:10px;background:var(--panel2);color:var(--text);cursor:pointer;font-size:20px}.side-option{width:100%;display:flex;align-items:center;gap:12px;padding:14px 12px;border:0;border-radius:11px;background:transparent;color:var(--text);font-size:16px;text-align:left;cursor:pointer}.side-option .arrow{margin-left:auto;font-size:20px}.side-option:hover{background:var(--panel2);color:var(--teal)}.theme-options{display:none;padding:5px 0 8px 12px;gap:6px}.theme-options.show{display:flex;flex-direction:column}.theme-options button{border:1px solid var(--border);border-radius:9px;background:var(--panel2);color:var(--text);padding:10px 12px;text-align:left;cursor:pointer;font-size:14px}.theme-options button:hover{color:var(--teal);border-color:var(--teal)}
.overlay{position:fixed;inset:0;z-index:40;background:rgba(0,0,0,.45);opacity:0;visibility:hidden;transition:.2s}.overlay.show{opacity:1;visibility:visible}
.about-overlay{position:fixed;inset:0;z-index:60;display:flex;align-items:center;justify-content:center;padding:20px;background:rgba(0,0,0,.5);opacity:0;visibility:hidden;transition:.2s}.about-overlay.show{opacity:1;visibility:visible}.about-card{position:relative;width:min(560px,100%);padding:34px;border-radius:22px;background:var(--panel);border:1px solid var(--border);box-shadow:0 30px 80px rgba(0,0,0,.25)}.about-close{position:absolute;right:18px;top:18px}.about-icon{width:52px;height:52px;border-radius:15px;display:flex;align-items:center;justify-content:center;background:var(--panel2);color:var(--teal);font-size:24px;margin-bottom:20px}.about-card h2{margin:0 0 14px;color:var(--text);font-size:28px}.about-card p{color:var(--muted);line-height:1.65;font-size:15px}.about-ok{margin-top:8px;border:0;border-radius:10px;padding:11px 18px;background:var(--teal);color:#fff;font-weight:700;cursor:pointer}
@media(max-width:700px){.topbar{padding:16px 18px}.brand{font-size:20px}.main-content{width:min(100% - 28px,980px);padding-top:40px}.dash-card{padding:24px}.sidebar{width:min(320px,88vw)}}
</style>
</head>
<body>

<div class="app-shell">

    <header class="topbar">
        <div>
            <div class="brand">Face Attendance System</div>
            <div class="tagline">Smart, simple and secure attendance management</div>
        </div>

        <button class="menu-btn" id="menuBtn" aria-label="Open dashboard menu">☰</button>
    </header>

    <!-- ONLY About + Theme are inside the hamburger menu -->
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="side-title">Dashboard</div>
            <button class="close-btn" id="closeBtn">×</button>
        </div>

        <div class="side-section">
            <button class="side-option" id="aboutBtn">
                <span>ⓘ</span>
                <span>About</span>
                <span class="arrow">›</span>
            </button>

            <div class="theme-section">
                <button class="side-option" id="themeToggle">
                    <span>◐</span>
                    <span>Theme</span>
                    <span class="arrow" id="themeArrow">›</span>
                </button>

                <div class="theme-options" id="themeOptions">
                    <button data-theme="light">☀ &nbsp; Light</button>
                    <button data-theme="system">◉ &nbsp; System</button>
                    <button data-theme="dark">☾ &nbsp; Dark</button>
                </div>
            </div>
        </div>
    </aside>

    <div class="overlay" id="overlay"></div>

    <!-- About popup -->
    <div class="about-overlay" id="aboutOverlay">
        <div class="about-card">
            <button class="about-close" id="aboutClose">×</button>
            <div class="about-icon">◉</div>
            <div class="eyebrow">ABOUT THE SYSTEM</div>
            <h2>Face Attendance System</h2>
            <p>
                This system uses face detection and face recognition to identify
                registered students and simplify attendance management.
            </p>
            <p>
                Students can be registered by capturing face samples. During
                attendance, the camera recognizes registered faces and displays
                the result. Attendance is marked only after explicit confirmation.
            </p>
            <button class="about-ok" id="aboutOk">Got it</button>
        </div>
    </div>

    <main class="main-content">

        <section class="hero">
            <div class="eyebrow">ATTENDANCE MANAGEMENT</div>
            <h1>Welcome to your <span>smart attendance dashboard</span></h1>
            <p>
                Choose an action below to manage student registration,
                face recognition, attendance and records.
            </p>
        </section>

        <!-- ALL MAIN ACTIONS STAY IN THE MIDDLE -->
        <section class="dashboard-grid">

            <article class="dash-card">
                <div class="card-icon">＋</div>
                <h2>Register Student</h2>
                <p>Add a new student and capture face samples for recognition.</p>
                <a class="primary-btn" href="/register">
                    Register Student <span>→</span>
                </a>
            </article>

            <article class="dash-card">
                <div class="card-icon">◉</div>
                <h2>Take &amp; Mark Attendance</h2>
                <p>Use the camera to recognize a student, then explicitly mark attendance.</p>
                <a class="primary-btn" href="/attendance">
                    Take Attendance <span>→</span>
                </a>
            </article>

            <article class="dash-card">
                <div class="card-icon">▤</div>
                <h2>Attendance Records</h2>
                <p>View today's attendance records and marking status.</p>
                <a class="primary-btn" href="/records">
                    View Records <span>→</span>
                </a>
            </article>

        </section>
    </main>

    <footer>
        Face Attendance System &nbsp;•&nbsp; Smart attendance management
    </footer>

</div>

<script>
(function () {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("overlay");
    const menuBtn = document.getElementById("menuBtn");
    const closeBtn = document.getElementById("closeBtn");
    const aboutBtn = document.getElementById("aboutBtn");
    const aboutOverlay = document.getElementById("aboutOverlay");
    const aboutClose = document.getElementById("aboutClose");
    const aboutOk = document.getElementById("aboutOk");
    const themeToggle = document.getElementById("themeToggle");
    const themeOptions = document.getElementById("themeOptions");
    const themeArrow = document.getElementById("themeArrow");

    function getTheme() {
        return localStorage.getItem("attendanceTheme") || "system";
    }

    function applyTheme(theme) {
        if (!["light", "dark", "system"].includes(theme)) theme = "system";
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("attendanceTheme", theme);
    }

    // Apply saved theme immediately. It survives every page navigation.
    applyTheme(getTheme());

    function openMenu() {
        sidebar.classList.add("open");
        overlay.classList.add("show");
    }

    function closeMenu() {
        sidebar.classList.remove("open");
        overlay.classList.remove("show");
    }

    menuBtn.addEventListener("click", openMenu);
    closeBtn.addEventListener("click", closeMenu);
    overlay.addEventListener("click", closeMenu);

    aboutBtn.addEventListener("click", function () {
        aboutOverlay.classList.add("show");
    });

    function closeAbout() {
        aboutOverlay.classList.remove("show");
    }

    aboutClose.addEventListener("click", closeAbout);
    aboutOk.addEventListener("click", closeAbout);
    aboutOverlay.addEventListener("click", function (event) {
        if (event.target === aboutOverlay) closeAbout();
    });

    // Theme is an expandable section. Clicking it NEVER changes the theme.
    themeToggle.addEventListener("click", function () {
        const open = themeOptions.classList.toggle("show");
        themeArrow.textContent = open ? "⌄" : "›";
    });

    // Only the three actual theme choices change the theme.
    themeOptions.querySelectorAll("button[data-theme]").forEach(function (button) {
        button.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();
            applyTheme(button.getAttribute("data-theme"));
            themeOptions.classList.remove("show");
            themeArrow.textContent = "›";
        });
    });
})();
</script>

</body>
</html>
"""



# PATHS
# =================================================

MODEL_PATH = "face_detection_yunet_2023mar.onnx"
DATASET_PATH = "dataset"

ATTENDANCE_FOLDER = "attendance"
ATTENDANCE_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "attendance.csv"
)


# =================================================
# CREATE ATTENDANCE FOLDER / FILE
# =================================================

os.makedirs(
    ATTENDANCE_FOLDER,
    exist_ok=True
)

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
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


# =================================================
# LOAD YUNET FACE DETECTOR
# =================================================

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 240),
    0.9,
    0.3,
    5000
)


# =================================================
# LOAD REGISTERED STUDENTS
# =================================================

print()
print("========================================")
print("Loading registered students...")
print("========================================")

known_faces = load_known_faces()

print()


# =================================================
# HOME PAGE
# =================================================

@app.route("/")
def home():
    return render_template_string(DASHBOARD_HTML)


# =================================================
# ATTENDANCE PAGE
# =================================================

@app.route("/attendance")
def attendance():

    return render_template(
        "attendance.html"
    )


# =================================================
# REGISTER PAGE
# =================================================

@app.route("/register")
def register():

    return render_template(
        "register.html"
    )


# =================================================
# CAPTURE FACE FOR REGISTRATION
# =================================================

@app.route(
    "/capture_face",
    methods=["POST"]
)
def capture_face():

    try:

        data = request.get_json()

        name = data.get(
            "name",
            ""
        ).strip()

        image_data = data.get(
            "image",
            ""
        )

        # -----------------------------------------
        # CHECK NAME
        # -----------------------------------------

        if not name:

            return jsonify({
                "success": False,
                "message": "Name is required."
            })


        # -----------------------------------------
        # CHECK IMAGE
        # -----------------------------------------

        if not image_data:

            return jsonify({
                "success": False,
                "message": "No image received."
            })


        # -----------------------------------------
        # DECODE IMAGE
        # -----------------------------------------

        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(
            image_data
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({
                "success": False,
                "message": "Could not process image."
            })


        # -----------------------------------------
        # DETECT FACE
        # -----------------------------------------

        height, width = frame.shape[:2]

        face_detector.setInputSize(
            (width, height)
        )

        _, faces = face_detector.detect(
            frame
        )


        if faces is None or len(faces) == 0:

            return jsonify({
                "success": False,
                "message": "No face detected."
            })


        # -----------------------------------------
        # FIRST FACE
        # -----------------------------------------

        face = faces[0]

        x, y, w, h = face[:4].astype(int)

        x = max(0, x)
        y = max(0, y)

        w = min(
            w,
            width - x
        )

        h = min(
            h,
            height - y
        )


        if w <= 0 or h <= 0:

            return jsonify({
                "success": False,
                "message": "Invalid face detected."
            })


        # -----------------------------------------
        # CROP FACE
        # -----------------------------------------

        face_image = frame[
            y:y+h,
            x:x+w
        ]


        if face_image.size == 0:

            return jsonify({
                "success": False,
                "message": "Could not crop face."
            })


        # -----------------------------------------
        # CREATE STUDENT FOLDER
        # -----------------------------------------

        save_path = os.path.join(
            DATASET_PATH,
            name
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )


        # -----------------------------------------
        # COUNT IMAGES
        # -----------------------------------------

        existing_files = [

            file

            for file in os.listdir(
                save_path
            )

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]

        count = len(
            existing_files
        )


        # -----------------------------------------
        # MAX 30 IMAGES
        # -----------------------------------------

        if count >= 30:

            return jsonify({

                "success": True,

                "complete": True,

                "count": 30,

                "message":
                    "Registration complete!"

            })


        # -----------------------------------------
        # SAVE IMAGE
        # -----------------------------------------

        filename = os.path.join(
            save_path,
            f"{name}_{count + 1}.jpg"
        )

        cv2.imwrite(
            filename,
            face_image
        )

        count += 1


        return jsonify({

            "success": True,

            "complete": count >= 30,

            "count": count,

            "message":
                f"Captured {count}/30"

        })


    except Exception as e:

        print(
            "Registration error:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                "Server error."

        })


# =================================================
# CHECK IF ATTENDANCE ALREADY EXISTS TODAY
# =================================================

def attendance_already_marked(name):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    if not os.path.exists(
        ATTENDANCE_FILE
    ):
        return False


    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                row_name = row.get(
                    "Name",
                    ""
                ).strip()

                row_date = row.get(
                    "Date",
                    ""
                ).strip()


                if (
                    row_name.lower() == name.lower()
                    and row_date == today
                ):

                    return True


    except Exception as e:

        print(
            "Attendance check error:",
            e
        )


    return False


# =================================================
# MARK ATTENDANCE
# =================================================

def mark_attendance(name):

    now = datetime.now()

    date = now.strftime(
        "%Y-%m-%d"
    )

    time = now.strftime(
        "%H:%M:%S"
    )


    # -----------------------------------------
    # DON'T DUPLICATE
    # -----------------------------------------

    if attendance_already_marked(
        name
    ):

        return False


    # -----------------------------------------
    # SAVE ATTENDANCE
    # -----------------------------------------

    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            date,
            time,
            "Present"
        ])


    print(
        f"Attendance marked: {name} "
        f"| {date} | {time}"
    )


    return True


# =================================================
# RECOGNIZE FACE
# IMPORTANT:
# THIS DOES NOT MARK ATTENDANCE
# =================================================

@app.route(
    "/recognize",
    methods=["POST"]
)
def recognize():

    global known_faces

    try:

        data = request.get_json()

        image_data = data.get(
            "image",
            ""
        )


        # -----------------------------------------
        # CHECK IMAGE
        # -----------------------------------------

        if not image_data:

            return jsonify({

                "success": False,

                "message":
                    "No image received.",

                "results": []

            })


        # -----------------------------------------
        # DECODE IMAGE
        # -----------------------------------------

        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(
            image_data
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({

                "success": False,

                "message":
                    "Could not process image.",

                "results": []

            })


        # -----------------------------------------
        # LOAD REGISTERED FACES
        # -----------------------------------------

        known_faces = load_known_faces()


        if len(known_faces) == 0:

            return jsonify({

                "success": False,

                "message":
                    "No registered students found.",

                "results": []

            })


        # -----------------------------------------
        # DETECT FACES
        # -----------------------------------------

        height, width = frame.shape[:2]

        face_detector.setInputSize(
            (width, height)
        )

        _, faces = face_detector.detect(
            frame
        )


        if faces is None or len(faces) == 0:

            return jsonify({

                "success": True,

                "message":
                    "No face detected.",

                "results": []

            })


        results = []


        # -----------------------------------------
        # RECOGNIZE EACH FACE
        # -----------------------------------------

        for face in faces:

            name, score = recognize_face(
                frame,
                face,
                known_faces
            )

            score = float(score)


            # =====================================
            # RECOGNIZED
            # =====================================

            if name != "Unknown":

                # IMPORTANT:
                # ONLY recognize the person.
                # DO NOT mark attendance here.

                already_marked = attendance_already_marked(
                    name
                )


                if already_marked:

                    status = (
                        "Attendance already marked today."
                    )

                else:

                    status = (
                        "Face recognized. "
                        "Ready to mark attendance."
                    )


                results.append({

                    "name": name,

                    "score": round(
                        score,
                        2
                    ),

                    "status": status

                })


            # =====================================
            # UNKNOWN
            # =====================================

            else:

                results.append({

                    "name": "Unknown",

                    "score": round(
                        score,
                        2
                    ),

                    "status":
                        "Not registered"

                })


        # -----------------------------------------
        # RESPONSE
        # -----------------------------------------

        return jsonify({

            "success": True,

            "message":
                "Face recognition complete.",

            "results": results

        })


    except Exception as e:

        print(
            "Recognition error:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                "Recognition error.",

            "results": []

        })


# =================================================
# EXPLICIT MARK ATTENDANCE ROUTE
# =================================================

@app.route(
    "/mark_attendance",
    methods=["POST"]
)
def mark_attendance_route():

    global known_faces

    try:

        data = request.get_json()

        name = data.get(
            "name",
            ""
        ).strip()


        # -----------------------------------------
        # CHECK NAME
        # -----------------------------------------

        if not name:

            return jsonify({

                "success": False,

                "message":
                    "Student name is required."

            })


        # -----------------------------------------
        # RELOAD REGISTERED STUDENTS
        # -----------------------------------------

        known_faces = load_known_faces()


        # -----------------------------------------
        # FIND ACTUAL REGISTERED NAME
        # -----------------------------------------

        registered_name = None

        for student_name in known_faces:

            if (
                student_name.lower()
                == name.lower()
            ):

                registered_name = student_name

                break


        # -----------------------------------------
        # STUDENT NOT REGISTERED
        # -----------------------------------------

        if registered_name is None:

            return jsonify({

                "success": False,

                "message":
                    "Student is not registered."

            })


        # -----------------------------------------
        # CHECK DUPLICATE
        # -----------------------------------------

        if attendance_already_marked(
            registered_name
        ):

            return jsonify({

                "success": True,

                "marked": False,

                "message":
                    "Attendance already marked today."

            })


        # -----------------------------------------
        # MARK ATTENDANCE
        # -----------------------------------------

        mark_attendance(
            registered_name
        )


        return jsonify({

            "success": True,

            "marked": True,

            "message":
                "Attendance marked successfully!"

        })


    except Exception as e:

        print(
            "Mark attendance error:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                "Could not mark attendance."

        })


@app.after_request
def add_global_style(response):
    if response.content_type and "text/html" in response.content_type:
        html = response.get_data(as_text=True)

        css = r"""<style>
:root{
    --bg:#e8f8f6;--panel:#ffffff;--panel2:#f1fbfa;--text:#173637;
    --muted:#647777;--teal:#087f7a;--teal2:#0b918b;--border:#cce7e4;
    --shadow:0 18px 45px rgba(8,78,75,.12)
}
:root[data-theme="dark"]{
    --bg:#071918;--panel:#102625;--panel2:#15302e;--text:#e9fffc;
    --muted:#a7c2c0;--teal:#5ee8dc;--teal2:#7af0e7;--border:#244844;
    --shadow:0 18px 45px rgba(0,0,0,.32)
}
@media (prefers-color-scheme:dark){
    :root[data-theme="system"]{
        --bg:#071918;--panel:#102625;--panel2:#15302e;--text:#e9fffc;
        --muted:#a7c2c0;--teal:#5ee8dc;--teal2:#7af0e7;--border:#244844;
        --shadow:0 18px 45px rgba(0,0,0,.32)
    }
}
html,body{background:var(--bg)!important;color:var(--text)}
body{font-family:"Segoe UI",Arial,sans-serif}

/* Keep all non-dashboard pages visually consistent. */
a[href="/"],a[href="/attendance"].dashboard-btn,a.dashboard-btn,
.back-btn,.go-dashboard,.dashboard-button{
    display:inline-flex!important;align-items:center;justify-content:center;gap:8px;
    padding:12px 20px!important;border-radius:12px!important;
    background:var(--teal)!important;background-image:none!important;
    color:#fff!important;-webkit-text-fill-color:#fff!important;
    border:0!important;border-width:0!important;outline:0!important;
    text-decoration:none!important;text-shadow:none!important;
    -webkit-appearance:none!important;appearance:none!important;
    font-weight:700!important;box-shadow:0 7px 18px rgba(8,127,122,.18)!important;
    transition:.2s ease!important
}
a[href="/"] *,a[href="/attendance"].dashboard-btn *,a.dashboard-btn *,
.back-btn *,.go-dashboard *,.dashboard-button *{
    background:transparent!important;background-image:none!important;
    color:#fff!important;-webkit-text-fill-color:#fff!important;
    border:0!important;outline:0!important;text-decoration:none!important;
}
a[href="/"]:hover,a[href="/attendance"].dashboard-btn:hover,a.dashboard-btn:hover,
.back-btn:hover,.go-dashboard:hover,.dashboard-button:hover{
    background:var(--teal2)!important;transform:translateY(-1px)
}

/* Make common page panels follow the selected theme. */
.card,.container,.form-container,.attendance-container,.register-container,
.records-card,.records-card-large,.panel,.box{
    background:var(--panel)!important;color:var(--text)!important;
    border-color:var(--border)!important
}

/* Dashboard */
.app-shell{min-height:100vh;background:linear-gradient(135deg,#e8faf7,#f9ffff)}
:root[data-theme="dark"] .app-shell{background:linear-gradient(135deg,#071918,#0c2220)}
@media(prefers-color-scheme:dark){:root[data-theme="system"] .app-shell{background:linear-gradient(135deg,#071918,#0c2220)}}
.topbar{background:rgba(255,255,255,.9)}
:root[data-theme="dark"] .topbar{background:#0a1f1e}
@media(prefers-color-scheme:dark){:root[data-theme="system"] .topbar{background:#0a1f1e}}
.menu-btn{background:var(--panel)!important;color:var(--teal)!important;border-color:var(--border)!important}
.sidebar{background:var(--panel)!important;border-color:var(--border)!important}
.side-option{color:var(--text)!important}
.side-option:hover{background:var(--panel2)!important;color:var(--teal)!important}
.theme-options button{color:var(--muted)!important}
.theme-options button:hover{background:var(--panel2)!important;color:var(--teal)!important}
.dash-card{background:var(--panel)!important;color:var(--text)!important;border-color:var(--border)!important}
.primary-btn{background:var(--teal)!important;color:#fff!important;border:0!important;border-color:transparent!important;outline:0!important}
.primary-btn:hover{background:var(--teal2)!important}

/* About */
.about-card{background:var(--panel)!important;color:var(--text)!important;border-color:var(--border)!important}
.about-card p{color:var(--muted)!important}
.about-close{background:var(--panel2)!important;color:var(--text)!important}
.about-ok{background:var(--teal)!important}

/* Records page */
.records-page{min-height:100vh;background:var(--bg)!important;color:var(--text)!important;padding:55px 34px}
.records-header{max-width:1050px;margin:0 auto 28px;display:flex;align-items:center;justify-content:space-between;gap:20px}
.records-header h1{margin:0 0 8px;color:var(--text)}
.records-header p{margin:0;color:var(--muted)}
.records-card-large{max-width:1050px;margin:auto;border:1px solid var(--border);border-radius:20px;overflow:hidden;box-shadow:var(--shadow)}
.records-card-large table{width:100%;border-collapse:collapse}
.records-card-large th,.records-card-large td{padding:16px 18px;text-align:left;border-bottom:1px solid var(--border)}
.records-card-large th{background:var(--panel2);color:var(--text)}
.records-card-large td{color:var(--text)}
.records-card-large .empty{text-align:center;color:var(--muted);padding:35px}

@media(max-width:760px){
    .records-page{padding:35px 18px}.records-header{flex-direction:column;align-items:flex-start}
}
</style>"""

        # Only add the theme initializer on pages that don't already contain it.
        theme_init = r"""<script>
(function(){
    try {
        var theme = localStorage.getItem('attendanceTheme') || 'system';
        if (['light','dark','system'].indexOf(theme) === -1) theme = 'system';
        document.documentElement.setAttribute('data-theme', theme);
    } catch(e) {
        document.documentElement.setAttribute('data-theme', 'system');
    }
})();
</script>"""

        if '</head>' in html:
            html = html.replace('</head>', css + theme_init + '</head>', 1)
        else:
            html = css + theme_init + html

        response.set_data(html)
    return response


# =================================================
# =================================================
# ATTENDANCE RECORDS
# =================================================
@app.route("/records")
def records():
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, "r", newline="") as file:
                for row in csv.DictReader(file):
                    if row.get("Date", "").strip() == today:
                        rows.append(row)
        except Exception as e:
            print("Records error:", e)
    table_rows = "".join("<tr><td>" + r.get("Name", "") + "</td><td>" + r.get("Date", "") + "</td><td>" + r.get("Time", "") + "</td><td>" + r.get("Status", "") + "</td></tr>" for r in rows)
    if not table_rows:
        table_rows = "<tr><td colspan='4' class='empty'>No attendance has been marked today.</td></tr>"
    page = """<!DOCTYPE html><html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1.0'><title>Attendance Records</title></head><body><div class='records-page'><div class='records-header'><div><div class='eyebrow'>TODAY'S ATTENDANCE</div><h1>Attendance Records</h1><p>""" + today + """</p></div><a class='primary-btn' href='/'>← Dashboard</a></div><div class='records-card-large'><table><thead><tr><th>Name</th><th>Date</th><th>Time</th><th>Status</th></tr></thead><tbody>""" + table_rows + """</tbody></table></div></div></body></html>"""
    return render_template_string(page)

# RUN FLASK
# =================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )