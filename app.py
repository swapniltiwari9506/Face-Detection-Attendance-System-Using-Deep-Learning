"""
Face Recognition Attendance System - single-file backend with login and roles.

Run:   pip install -r requirements.txt
       python app.py
Open:  http://127.0.0.1:8000

Access
  guest   - sees the page and the tab names, but opening Students / Attendance /
            Reports sends them straight to the login/sign-up page. No data is
            ever returned without a session (every /api route checks it).

Roles
  admin   - everything, plus the Users tab (change roles, delete accounts)
  teacher - Students, Attendance, Reports, face recognition dashboard
  student - own dashboard (camera marks ONLY their own face) and own attendance

The very first account created on an empty database becomes the admin.
Teachers sign up with the STAFF_CODE below (change it, or set the STAFF_CODE
environment variable). Students sign up with their registered Student ID.
"""
import base64
import csv
import hashlib
import hmac
import io
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
import urllib.request
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = DATA / "models"
PHOTOS = DATA / "student_images"
DB_PATH = DATA / "attendance.db"
INDEX_HTML = BASE / "index.html"
LOGIN_HTML = BASE / "login.html"

HOST, PORT = "127.0.0.1", 8000
MATCH_THRESHOLD = 0.363      # SFace cosine similarity: higher = stricter
DUPLICATE_THRESHOLD = 0.55   # block registering the same face under two IDs
MAX_IMAGE_SIDE = 1280
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

# --- auth settings ---
STAFF_CODE = os.environ.get("STAFF_CODE", "TEACHER2026")   # needed to sign up as a teacher
COOKIE_NAME = "fa_session"
SESSION_DAYS = 7
PBKDF2_ITERS = 200_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
ROLES = ("admin", "teacher", "student")
STAFF_ROLES = ("admin", "teacher")

# filename -> (exact expected size in bytes, [download URLs tried in order])
MODEL_FILES = {
    "yunet.onnx": (
        232589,
        [
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    ),
    "sface.onnx": (
        38696353,
        [
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    ),
}

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Asia/Kolkata")
except Exception:  # Windows without tzdata
    TZ = timezone(timedelta(hours=5, minutes=30))


def now():
    return datetime.now(TZ)


def now_str():
    return now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------
detector = None
recognizer = None
cv_lock = threading.Lock()      # OpenCV detector is not thread-safe
cache_lock = threading.Lock()
EMB = np.zeros((0, 128), np.float32)
EMB_IDS: List[str] = []


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                vec        BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name       TEXT NOT NULL,
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'Present',
                UNIQUE (student_id, date)
            );
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL CHECK (role IN ('admin','teacher','student')),
                student_id    TEXT UNIQUE,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                expires_at TEXT NOT NULL
            );
            """
        )


def load_cache():
    global EMB, EMB_IDS
    with db() as c:
        rows = c.execute("SELECT student_id, vec FROM embeddings").fetchall()
    with cache_lock:
        if rows:
            EMB = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
            EMB_IDS = [r["student_id"] for r in rows]
        else:
            EMB = np.zeros((0, 128), np.float32)
            EMB_IDS = []


# ----------------------------------------------------------------------------
# Authentication helpers
# ----------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERS)
    return f"pbkdf2${PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt, dk = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(calc.hex(), dk)
    except Exception:
        return False


# Used so a wrong username costs the same time as a wrong password.
_DUMMY_HASH = hash_password("not-a-real-password")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = (now() + timedelta(days=SESSION_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    with db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now_str(),))
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (hash_token(token), user_id, expires),
        )
    return token


def set_session_cookie(response: Response, token: str):
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax", path="/",
    )


def user_from_request(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    with db() as c:
        row = c.execute(
            """
            SELECT u.id, u.username, u.name, u.role, u.student_id
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ?
            """,
            (hash_token(token), now_str()),
        ).fetchone()
    return dict(row) if row else None


def current_user(request: Request) -> dict:
    user = user_from_request(request)
    if not user:
        raise HTTPException(401, "Please log in")
    return user


def staff_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] not in STAFF_ROLES:
        raise HTTPException(403, "Only teachers and admins can do this")
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Only admins can do this")
    return user


# Simple brute-force protection: 5 failed logins per (ip, username) per 5 minutes.
_FAILS: dict = {}
_fail_lock = threading.Lock()


def _recent_fails(key: str):
    t = time.time()
    lst = [x for x in _FAILS.get(key, []) if t - x < 300]
    _FAILS[key] = lst
    return lst


def login_blocked(key: str) -> bool:
    with _fail_lock:
        return len(_recent_fails(key)) >= 5


def record_fail(key: str):
    with _fail_lock:
        _recent_fails(key).append(time.time())


def clear_fails(key: str):
    with _fail_lock:
        _FAILS.pop(key, None)


# ----------------------------------------------------------------------------
# Models / face helpers
# ----------------------------------------------------------------------------
def download_models():
    """Download models, verify exact size, and replace corrupt/partial files."""
    MODELS.mkdir(parents=True, exist_ok=True)
    for filename, (expected, urls) in MODEL_FILES.items():
        path = MODELS / filename
        if path.exists() and path.stat().st_size == expected:
            print(f"  {filename}: OK ({expected} bytes)")
            continue
        if path.exists():
            print(f"  {filename}: wrong size ({path.stat().st_size} bytes), downloading again...")
            path.unlink()
        ok = False
        for url in urls:
            for attempt in (1, 2):
                tmp = path.with_suffix(".part")
                try:
                    print(f"Downloading {filename} (attempt {attempt}) ...")
                    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                        shutil.copyfileobj(r, f)
                    if tmp.stat().st_size == expected:
                        tmp.replace(path)
                        ok = True
                        break
                    print(f"  incomplete download: {tmp.stat().st_size} of {expected} bytes")
                except Exception as e:
                    print(f"  download failed: {e}")
                finally:
                    if tmp.exists() and not ok:
                        tmp.unlink()
            if ok:
                break
        if not ok:
            raise RuntimeError(
                f"Could not download {filename} correctly. Check your internet connection and run again, "
                f"or download it manually from:\n  {urls[0]}\nand save it as {path}"
            )
        print(f"  {filename}: downloaded ({expected} bytes)")


def load_models():
    global detector, recognizer
    detector = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (320, 320), 0.80, 0.30, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(MODELS / "sface.onnx"), "")


def decode_image(b64: str):
    if "," in b64[:100]:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Image is not valid Base64 data")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the image")
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_faces(img):
    with cv_lock:
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
    return [] if faces is None else faces


def embed(img, face):
    with cv_lock:
        aligned = recognizer.alignCrop(img, face)
        feature = recognizer.feature(aligned).flatten().astype(np.float32)
    norm = np.linalg.norm(feature)
    return None if norm == 0 else feature / norm


def best_match(query):
    """Return (student_id, similarity) of the closest stored face, or (None, 0)."""
    with cache_lock:
        emb, ids = EMB, EMB_IDS
    if len(ids) == 0:
        return None, 0.0
    scores = emb @ query
    i = int(np.argmax(scores))
    return ids[i], float(scores[i])


def check_id(student_id: str):
    if not ID_PATTERN.match(student_id or ""):
        raise HTTPException(400, "Student ID may contain only letters, digits, _ and - (max 32 chars)")


def check_date(date: str):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Date must be in YYYY-MM-DD format")


# ----------------------------------------------------------------------------
# API models
# ----------------------------------------------------------------------------
class FramePayload(BaseModel):
    image_base64: str


class RegisterPayload(BaseModel):
    student_id: str
    name: str
    images: List[str]


class ManualPayload(BaseModel):
    student_id: str
    date: Optional[str] = None


class SignupPayload(BaseModel):
    name: str
    username: str
    password: str
    role: str = "student"
    student_id: Optional[str] = None
    staff_code: Optional[str] = None


class LoginPayload(BaseModel):
    username: str
    password: str


class RolePayload(BaseModel):
    role: str


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA.mkdir(exist_ok=True)
    PHOTOS.mkdir(parents=True, exist_ok=True)
    download_models()
    load_models()
    init_db()
    load_cache()
    url = f"http://{HOST}:{PORT}"
    print(f"\nFace Attendance System running at {url}")
    print(f"Teacher sign-up code: {STAFF_CODE}   (change it with the STAFF_CODE environment variable)\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    yield


app = FastAPI(title="Face Attendance System", lifespan=lifespan)


def html_page(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found next to app.py")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


TAB_NAMES = ("dashboard", "mine", "students", "attendance", "reports", "users")


@app.get("/")
def home():
    # The page shell is public. It contains NO data: every /api route behind it
    # needs a valid session, and the page sends guests to /login as soon as they
    # open a protected tab (Students, Attendance, Reports).
    return html_page(INDEX_HTML)


@app.get("/login")
def login_page(request: Request, next: Optional[str] = None):
    if user_from_request(request):
        target = "/#" + next if next in TAB_NAMES else "/"
        return RedirectResponse(target, status_code=303)
    return html_page(LOGIN_HTML)


# ------------------------------- auth API -----------------------------------
def public_user(row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "name": row["name"],
        "role": row["role"],
        "student_id": row["student_id"],
    }


@app.get("/api/auth/status")
def api_auth_status():
    with db() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    return {"has_users": n > 0}


@app.post("/api/auth/signup")
def api_signup(payload: SignupPayload, response: Response):
    name = payload.name.strip()
    username = payload.username.strip().lower()
    password = payload.password

    if not name or len(name) > 60:
        raise HTTPException(400, "Enter your full name (max 60 characters)")
    if not USERNAME_PATTERN.match(username):
        raise HTTPException(400, "Username must be 3-32 characters: letters, digits, . _ -")
    if len(password) < 8 or len(password) > 128:
        raise HTTPException(400, "Password must be 8 to 128 characters")
    if payload.role not in ("student", "teacher"):
        raise HTTPException(400, "Choose Student or Teacher")

    role = payload.role
    student_id = None

    with db() as c:
        first_user = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0
        if first_user:
            role = "admin"                      # bootstrap: first account is the admin
        elif role == "teacher":
            if not hmac.compare_digest((payload.staff_code or "").encode(), STAFF_CODE.encode()):
                raise HTTPException(403, "Invalid staff access code")
        else:  # student
            student_id = (payload.student_id or "").strip()
            check_id(student_id)
            if c.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,)).fetchone() is None:
                raise HTTPException(
                    404, "This Student ID is not registered yet. Ask your teacher to register your face first."
                )
            if c.execute("SELECT 1 FROM users WHERE student_id = ?", (student_id,)).fetchone():
                raise HTTPException(409, "An account already exists for this Student ID")

        try:
            cur = c.execute(
                "INSERT INTO users (username, name, password_hash, role, student_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, name, hash_password(password), role, student_id, now_str()),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "That username is already taken")
        user_id = cur.lastrowid

    token = create_session(user_id)
    set_session_cookie(response, token)
    return {"status": "success", "user": {"id": user_id, "username": username, "name": name,
                                          "role": role, "student_id": student_id}}


@app.post("/api/auth/login")
def api_login(payload: LoginPayload, request: Request, response: Response):
    username = payload.username.strip().lower()
    ip = request.client.host if request.client else "?"
    key = f"{ip}|{username}"
    if login_blocked(key):
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.")

    with db() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    ok = verify_password(payload.password, row["password_hash"] if row else _DUMMY_HASH)
    if not row or not ok:
        record_fail(key)
        raise HTTPException(401, "Incorrect username or password")

    clear_fails(key)
    token = create_session(row["id"])
    set_session_cookie(response, token)
    return {"status": "success", "user": public_user(row)}


@app.post("/api/auth/logout")
def api_logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        with db() as c:
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"status": "success"}


@app.get("/api/auth/me")
def api_me(user: dict = Depends(current_user)):
    return user


# ------------------------------ admin: users --------------------------------
@app.get("/api/users")
def api_users(admin: dict = Depends(admin_user)):
    with db() as c:
        rows = c.execute(
            "SELECT id, username, name, role, student_id, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return {"users": [dict(r) for r in rows], "me": admin["id"]}


@app.put("/api/users/{user_id}/role")
def api_set_role(user_id: int, payload: RolePayload, admin: dict = Depends(admin_user)):
    if payload.role not in ROLES:
        raise HTTPException(400, "Invalid role")
    if user_id == admin["id"]:
        raise HTTPException(400, "You cannot change your own role")
    with db() as c:
        row = c.execute("SELECT student_id FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "User not found")
        if payload.role == "student" and not row["student_id"]:
            raise HTTPException(400, "This user has no linked Student ID, so they cannot be a student")
        c.execute("UPDATE users SET role = ? WHERE id = ?", (payload.role, user_id))
    return {"status": "success"}


@app.delete("/api/users/{user_id}")
def api_delete_user(user_id: int, admin: dict = Depends(admin_user)):
    if user_id == admin["id"]:
        raise HTTPException(400, "You cannot delete your own account")
    with db() as c:
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return {"status": "success"}


# ---------------------------- attendance helpers ----------------------------
def day_summary(date: str):
    with db() as c:
        students = c.execute("SELECT student_id, name FROM students ORDER BY name").fetchall()
        recs = c.execute(
            "SELECT student_id, name, date, time, status FROM attendance WHERE date = ? ORDER BY time",
            (date,),
        ).fetchall()
    current_ids = {s["student_id"] for s in students}
    records = [dict(r) for r in recs if r["student_id"] in current_ids]
    present_ids = {r["student_id"] for r in records}
    absent = [dict(s) for s in students if s["student_id"] not in present_ids]
    total = len(students)
    present = len(present_ids)
    return {
        "date": date,
        "total_students": total,
        "present_today": present,
        "absent_today": total - present,
        "attendance_percentage": round(present / total * 100) if total else 0,
        "attendance": records,
        "absent": absent,
    }


@app.get("/api/summary")
def api_summary(date: Optional[str] = None, user: dict = Depends(staff_user)):
    date = date or now().strftime("%Y-%m-%d")
    check_date(date)
    return day_summary(date)


@app.get("/api/attendance")
def api_attendance(date: Optional[str] = None, user: dict = Depends(staff_user)):
    date = date or now().strftime("%Y-%m-%d")
    check_date(date)
    return day_summary(date)


@app.get("/api/my/attendance")
def api_my_attendance(user: dict = Depends(current_user)):
    """A student's own attendance history. Only ever returns the caller's own data."""
    sid = user["student_id"]
    if not sid:
        raise HTTPException(403, "This account is not linked to a student")
    today = now().strftime("%Y-%m-%d")
    with db() as c:
        stu = c.execute("SELECT name FROM students WHERE student_id = ?", (sid,)).fetchone()
        samples = c.execute("SELECT COUNT(*) AS n FROM embeddings WHERE student_id = ?", (sid,)).fetchone()["n"]
        days = [r["date"] for r in c.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")]
        mine = {r["date"]: r["time"] for r in
                c.execute("SELECT date, time FROM attendance WHERE student_id = ?", (sid,))}
    total_days = len(days)
    present = sum(1 for d in days if d in mine)
    history = [
        {"date": d, "status": "Present" if d in mine else "Absent", "time": mine.get(d, "")}
        for d in days[:90]
    ]
    return {
        "student_id": sid,
        "registered": stu is not None,
        "face_samples": samples,
        "total_days": total_days,
        "present": present,
        "absent": total_days - present,
        "percentage": round(present / total_days * 100) if total_days else 0,
        "today_status": "Present" if today in mine else "Not marked yet",
        "today_time": mine.get(today, ""),
        "history": history,
    }


@app.post("/api/mark-attendance")
def api_mark_attendance(payload: FramePayload, user: dict = Depends(current_user)):
    with cache_lock:
        empty = len(EMB_IDS) == 0
    if empty:
        return {"status": "error", "message": "No students registered yet. Add a student first."}
    if user["role"] == "student" and not user["student_id"]:
        raise HTTPException(403, "This account is not linked to a student")

    frame = decode_image(payload.image_base64)
    faces = detect_faces(frame)
    if len(faces) == 0:
        return {"status": "no_face", "message": "No face detected"}
    if len(faces) > 1:
        return {"status": "multiple_faces", "message": "More than one face in the frame"}

    face = faces[0]
    x, y, w, h = [float(v) for v in face[:4]]
    box = {"x": x, "y": y, "w": w, "h": h, "fw": frame.shape[1], "fh": frame.shape[0]}

    query = embed(frame, face)
    if query is None:
        return {"status": "unknown", "message": "Could not compute face features", "box": box}

    student_id, similarity = best_match(query)
    if student_id is None or similarity < MATCH_THRESHOLD:
        return {
            "status": "unknown",
            "message": "Face did not match a registered student",
            "similarity": round(similarity, 3),
            "box": box,
        }

    # A logged-in student can only mark their own attendance (no proxy attendance).
    if user["role"] == "student" and student_id != user["student_id"]:
        return {"status": "not_you", "message": "This face does not match your account", "box": box}

    t = now()
    with db() as c:
        row = c.execute("SELECT name FROM students WHERE student_id = ?", (student_id,)).fetchone()
        if row is None:
            return {"status": "unknown", "message": "Matched student no longer exists", "box": box}
        name = row["name"]
        cur = c.execute(
            "INSERT OR IGNORE INTO attendance (student_id, name, date, time, status) VALUES (?, ?, ?, ?, 'Present')",
            (student_id, name, t.strftime("%Y-%m-%d"), t.strftime("%H:%M:%S")),
        )
        marked_now = cur.rowcount == 1

    return {
        "status": "success",
        "student_id": student_id,
        "name": name,
        "marked_now": marked_now,
        "similarity": round(similarity, 3),
        "message": f"Attendance marked for {name}" if marked_now else f"{name} is already marked present today",
        "box": box,
    }


@app.post("/api/attendance/manual")
def api_manual(payload: ManualPayload, user: dict = Depends(staff_user)):
    date = payload.date or now().strftime("%Y-%m-%d")
    check_date(date)
    with db() as c:
        s = c.execute("SELECT name FROM students WHERE student_id = ?", (payload.student_id,)).fetchone()
        if s is None:
            raise HTTPException(404, "Student not found")
        c.execute(
            "INSERT OR IGNORE INTO attendance (student_id, name, date, time, status) VALUES (?, ?, ?, ?, 'Present')",
            (payload.student_id, s["name"], date, now().strftime("%H:%M:%S")),
        )
    return {"status": "success"}


@app.delete("/api/attendance/{student_id}/{date}")
def api_delete_attendance(student_id: str, date: str, user: dict = Depends(staff_user)):
    check_date(date)
    with db() as c:
        c.execute("DELETE FROM attendance WHERE student_id = ? AND date = ?", (student_id, date))
    return {"status": "success"}


# ------------------------------- students ----------------------------------
@app.get("/api/students")
def api_students(user: dict = Depends(staff_user)):
    with db() as c:
        rows = c.execute(
            """
            SELECT s.student_id, s.name, s.created_at,
                   (SELECT COUNT(*) FROM embeddings e WHERE e.student_id = s.student_id) AS samples
            FROM students s ORDER BY s.name
            """
        ).fetchall()
    return {"students": [dict(r) for r in rows]}


@app.post("/api/students")
def api_register(payload: RegisterPayload, user: dict = Depends(staff_user)):
    student_id = payload.student_id.strip()
    name = payload.name.strip()
    check_id(student_id)
    if not name:
        raise HTTPException(400, "Name is required")
    if not payload.images:
        raise HTTPException(400, "Add at least one photo")
    if len(payload.images) > 15:
        raise HTTPException(400, "Maximum 15 photos at a time")

    folder = PHOTOS / student_id
    accepted, rejected = [], []

    for idx, b64 in enumerate(payload.images, start=1):
        img = decode_image(b64)
        faces = detect_faces(img)
        if len(faces) != 1:
            rejected.append(f"Photo {idx}: needs exactly one clear face (found {len(faces)})")
            continue
        vec = embed(img, faces[0])
        if vec is None:
            rejected.append(f"Photo {idx}: could not compute face features")
            continue
        other_id, sim = best_match(vec)
        if other_id is not None and other_id != student_id and sim >= DUPLICATE_THRESHOLD:
            rejected.append(f"Photo {idx}: this face already belongs to student {other_id}")
            continue
        accepted.append((img, vec))

    if not accepted:
        raise HTTPException(400, "No usable photos. " + " | ".join(rejected))

    folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(folder.glob("photo_*.jpg")))
    with db() as c:
        c.execute(
            "INSERT INTO students (student_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(student_id) DO UPDATE SET name = excluded.name",
            (student_id, name, now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        for n, (img, vec) in enumerate(accepted, start=existing + 1):
            cv2.imwrite(str(folder / f"photo_{n}.jpg"), img)
            c.execute("INSERT INTO embeddings (student_id, vec) VALUES (?, ?)", (student_id, vec.tobytes()))
    load_cache()
    return {"status": "success", "saved": len(accepted), "rejected": rejected}


@app.delete("/api/students/{student_id}")
def api_delete_student(student_id: str, user: dict = Depends(staff_user)):
    check_id(student_id)
    with db() as c:
        c.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))
        c.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    shutil.rmtree(PHOTOS / student_id, ignore_errors=True)
    load_cache()
    return {"status": "success"}


@app.get("/api/students/{student_id}/photo")
def api_student_photo(student_id: str, user: dict = Depends(current_user)):
    check_id(student_id)
    if user["role"] == "student" and user["student_id"] != student_id:
        raise HTTPException(403, "Not allowed")
    photos = sorted((PHOTOS / student_id).glob("photo_*.jpg"))
    if not photos:
        raise HTTPException(404, "No photo")
    return FileResponse(photos[0])


# ------------------------------ reports / export ---------------------------
@app.get("/api/report")
def api_report(user: dict = Depends(staff_user)):
    with db() as c:
        total_days = c.execute("SELECT COUNT(DISTINCT date) AS d FROM attendance").fetchone()["d"]
        rows = c.execute(
            """
            SELECT s.student_id, s.name, COUNT(a.id) AS present
            FROM students s LEFT JOIN attendance a ON a.student_id = s.student_id
            GROUP BY s.student_id, s.name ORDER BY s.name
            """
        ).fetchall()
    students = []
    for r in rows:
        pct = round(r["present"] / total_days * 100) if total_days else 0
        students.append({
            "student_id": r["student_id"],
            "name": r["name"],
            "present": r["present"],
            "absent": max(0, total_days - r["present"]),
            "percentage": pct,
        })
    return {"total_days": total_days, "students": students}


def csv_safe(value):
    """Stop spreadsheet apps from treating a cell as a formula."""
    s = "" if value is None else str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


@app.get("/api/export")
def api_export(date: Optional[str] = None, user: dict = Depends(staff_user)):
    query = "SELECT student_id, name, date, time, status FROM attendance"
    params = ()
    if date:
        check_date(date)
        query += " WHERE date = ?"
        params = (date,)
    query += " ORDER BY date, time"
    with db() as c:
        rows = c.execute(query, params).fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Student_ID", "Name", "Date", "Time", "Status"])
    for r in rows:
        writer.writerow([csv_safe(r["student_id"]), csv_safe(r["name"]), r["date"], r["time"], r["status"]])
    filename = f"attendance_{date or 'all'}.csv"
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")