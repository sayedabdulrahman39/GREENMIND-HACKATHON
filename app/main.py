import os
import time
import threading
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ---- Optional libs ----
# Webhook
requests_available = True
try:
    import requests
except Exception:
    requests_available = False

# SMS (Twilio)
twilio_available = True
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    twilio_available = False

# Firebase
firebase_available = True
try:
    import firebase_admin
    from firebase_admin import credentials, storage, firestore
except Exception:
    firebase_available = False

# ---- Load .env ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Firebase
FIREBASE_CRED = os.getenv("FIREBASE_CRED", "backend/.secrets/serviceAccountKey.json")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")

# Core config
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
try:
    CAMERA_SOURCE = int(CAMERA_SOURCE)
except Exception:
    pass

BRIGHTNESS_THRESHOLD = float(os.getenv("BRIGHTNESS_THRESHOLD", "70"))
ALERT_EMPTY_SECONDS = int(os.getenv("ALERT_EMPTY_SECONDS", "10"))
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "120"))

# Detection/perf
DETECTOR = os.getenv("DETECTOR", "motion").lower()   # motion | hog
FRAME_MAX = int(os.getenv("FRAME_MAX", "320"))
DETECT_EVERY_N = int(os.getenv("DETECT_EVERY_N", "6"))
CAM_FPS = int(os.getenv("CAM_FPS", "15"))
CAP_BACKEND = os.getenv("CAP_BACKEND", "msmf").lower()  # any | dshow | msmf
USE_MJPG = os.getenv("USE_MJPG", "0") == "1"
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "320"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "240"))
FLUSH_GRABS = int(os.getenv("FLUSH_GRABS", "2"))

# Clip/Snapshot
CLIP_BUFFER_SECONDS = int(os.getenv("CLIP_BUFFER_SECONDS", "6"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "12"))
STORE_VIDEO = os.getenv("STORE_VIDEO", "0") == "1"
STORE_SNAPSHOT = os.getenv("STORE_SNAPSHOT", "1") == "1"

# Webhook
HOST_WEBHOOK_URL = os.getenv("HOST_WEBHOOK_URL", "").strip()

# SMS (Twilio)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "").strip()

# Motion params (used if DETECTOR=motion)
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "800"))
MOTION_DILATE = int(os.getenv("MOTION_DILATE", "1"))
MOTION_SENSITIVITY = int(os.getenv("MOTION_SENSITIVITY", "3"))

# ---- Firebase init ----
db = None
bucket = None
firebase_ok = False
if firebase_available and FIREBASE_CRED and os.path.exists(FIREBASE_CRED):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED)
            firebase_admin.initialize_app(cred, {
                "projectId": FIREBASE_PROJECT_ID,
                "storageBucket": FIREBASE_STORAGE_BUCKET
            })
        db = firestore.client()
        bucket = storage.bucket()
        firebase_ok = True
        print("[Firebase] Initialized OK")
    except Exception as e:
        print("[Firebase] Init failed:", e)
else:
    print("[Firebase] Credentials not found. Running without Firebase.")

# ---- Twilio init ----
twilio_ok = False
twilio_client = None
if twilio_available and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and TWILIO_TO_NUMBER:
    try:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        twilio_ok = True
        print("[Twilio] Ready for SMS alerts")
    except Exception as e:
        print("[Twilio] Init failed:", e)

def backend_flag():
    if os.name != "nt":
        return 0
    if CAP_BACKEND == "dshow":
        return cv2.CAP_DSHOW
    if CAP_BACKEND == "msmf":
        return cv2.CAP_MSMF
    return 0

def send_webhook(payload: Dict[str, Any]):
    if not HOST_WEBHOOK_URL or not requests_available:
        return
    try:
        requests.post(HOST_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print("[Webhook] Failed:", e)

def send_sms(text: str):
    if not twilio_ok:
        print("[SMS] (dry-run) ->", text)
        return
    try:
        twilio_client.messages.create(
            body=text,
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER
        )
        print("[SMS] sent")
    except Exception as e:
        print("[SMS] failed:", e)

class CameraWorker(threading.Thread):
    def __init__(self, source=0):
        super().__init__(daemon=True)
        self.source = source
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "people_count": 0,
            "brightness": 0.0,
            "light_on": False,
            "last_alert_ts": None,
            "last_frame": None,   # ndarray; JPEG on-demand
            "last_alert_image_url": None,
            "last_alert_video_url": None,
            "last_error": None,
            "empty_since": None,
        }
        # Detectors
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=16 * MOTION_SENSITIVITY, detectShadows=False
        )

        self.frame_idx = 0
        self._last_people_count = 0

        # Ring buffer for pre-alert video (even if video off, keep tiny buffer for future)
        max_frames = max(1, CLIP_BUFFER_SECONDS * VIDEO_FPS)
        self.buffer = deque(maxlen=max_frames)

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source, backend_flag())

        if not self.cap.isOpened():
            with self.lock:
                self.state["last_error"] = f"Failed to open camera source: {self.source}"
            print("[Camera] Could not open source:", self.source)
            self.running = False
            return

        # Low latency hints
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            if USE_MJPG:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        last_alert_time = 0.0

        while self.running:
            # flush stale frames
            for _ in range(max(0, FLUSH_GRABS)):
                self.cap.grab()
            ok, frame = self.cap.retrieve()
            if not ok:
                ok, frame = self.cap.read()
                if not ok:
                    with self.lock:
                        self.state["last_error"] = "Failed to read frame"
                    time.sleep(0.01)
                    continue

            # Resize
            h, w = frame.shape[:2]
            scale = float(FRAME_MAX) / float(max(h, w))
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            now = time.time()
            self.buffer.append((now, frame_small.copy()))

            # Detection
            self.frame_idx += 1
            people_count = 0

            if DETECTOR == "motion":
                if self.frame_idx % max(1, DETECT_EVERY_N) == 0:
                    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                    fg = self.bg.apply(gray)
                    if MOTION_DILATE > 0:
                        fg = cv2.dilate(fg, None, iterations=MOTION_DILATE)
                        fg = cv2.erode(fg, None, iterations=1)
                    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pc = 0
                    for c in cnts:
                        if cv2.contourArea(c) >= MOTION_MIN_AREA:
                            pc += 1
                    people_count = int(min(pc, 5))
                    self._last_people_count = people_count
                else:
                    people_count = self._last_people_count

            elif DETECTOR == "hog":
                if self.frame_idx % max(1, DETECT_EVERY_N) == 0:
                    rects, _ = self.hog.detectMultiScale(
                        frame_small, winStride=(16, 16), padding=(8, 8), scale=1.2
                    )
                    people_count = len(rects)
                    self._last_people_count = people_count
                else:
                    people_count = self._last_people_count

            else:  # off
                people_count = 0

            # Brightness
            v = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)[2:, :, 2]
            brightness = float(np.mean(v))
            light_on = brightness >= BRIGHTNESS_THRESHOLD

            # Empty window logic
            with self.lock:
                prev_empty_since = self.state.get("empty_since")
                empty_since = prev_empty_since if (people_count == 0 and light_on) else None
                if people_count == 0 and light_on and prev_empty_since is None:
                    empty_since = now

            # Alert logic
            should_alert = False
            if people_count == 0 and light_on:
                if empty_since is None:
                    empty_since = now
                if (now - empty_since) >= ALERT_EMPTY_SECONDS and (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    should_alert = True

            alert_image_url = None
            alert_video_url = None
            if should_alert:
                ts_iso = datetime.now(timezone.utc).isoformat()
                ymd = datetime.now().strftime("%Y%m%d")
                hms = datetime.now().strftime("%H%M%S")

                # Snapshot (local encode; upload only if Firebase configured)
                if STORE_SNAPSHOT:
                    ret, jpeg = cv2.imencode(".jpg", frame_small)
                    if ret and firebase_ok and bucket is not None:
                        try:
                            img_path = f"alerts/{ymd}/{hms}_snap.jpg"
                            blob = bucket.blob(img_path)
                            blob.upload_from_string(jpeg.tobytes(), content_type="image/jpeg")
                            try:
                                blob.make_public()
                                alert_image_url = blob.public_url
                            except Exception:
                                alert_image_url = None
                        except Exception as e:
                            print("[Firebase] Storage upload (image) failed:", e)

                # Video clip (OFF for prototype unless STORE_VIDEO=1)
                if STORE_VIDEO:
                    cutoff = now - CLIP_BUFFER_SECONDS
                    frames = [fr for (t, fr) in list(self.buffer) if t >= cutoff]
                    if frames:
                        tmp_dir = tempfile.gettempdir()
                        vid_local = os.path.join(tmp_dir, f"alert_{ymd}_{hms}.avi")
                        try:
                            h0, w0 = frames[0].shape[:2]
                            writer = cv2.VideoWriter(
                                vid_local,
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                max(1, VIDEO_FPS),
                                (w0, h0)
                            )
                            for fr in frames:
                                writer.write(fr)
                            writer.release()
                            if firebase_ok and bucket is not None and os.path.exists(vid_local):
                                vid_path = f"alerts/{ymd}/{hms}_clip.avi"
                                blob = bucket.blob(vid_path)
                                blob.upload_from_filename(vid_local, content_type="video/avi")
                                try:
                                    blob.make_public()
                                    alert_video_url = blob.public_url
                                except Exception:
                                    alert_video_url = None
                        except Exception as e:
                            print("[Clip] Write/upload failed:", e)
                        finally:
                            try:
                                if os.path.exists(vid_local):
                                    os.remove(vid_local)
                            except Exception:
                                pass

                # Firestore doc (optional)
                if firebase_ok and db is not None:
                    try:
                        db.collection("alerts").add({
                            "message": "you forgot to turn off electricity",
                            "ts": ts_iso,
                            "people_count": int(people_count),
                            "brightness": float(brightness),
                            "light_on": bool(light_on),
                            "image_url": alert_image_url,
                            "video_url": alert_video_url,
                        })
                    except Exception as e:
                        print("[Firebase] Firestore write failed:", e)

                # SMS
                sms_text = "ALERT: you forgot to turn off electricity (room empty, light ON)."
                send_sms(sms_text)

                # Webhook (optional)
                send_webhook({
                    "event": "alert",
                    "message": "you forgot to turn off electricity",
                    "people_count": int(people_count),
                    "light_on": bool(light_on),
                    "brightness": float(brightness),
                    "image_url": alert_image_url,
                    "video_url": alert_video_url,
                    "timestamp": ts_iso,
                })

                last_alert_time = now

            # Save last frame & state
            with self.lock:
                self.state.update({
                    "people_count": int(people_count),
                    "brightness": round(brightness, 2),
                    "light_on": bool(light_on),
                    "last_alert_ts": last_alert_time if last_alert_time else None,
                    "last_frame": frame_small,
                    "last_alert_image_url": alert_image_url or self.state.get("last_alert_image_url"),
                    "last_alert_video_url": alert_video_url or self.state.get("last_alert_video_url"),
                    "empty_since": empty_since,
                    "last_error": None,
                })

            time.sleep(0.005)

        try:
            self.cap.release()
        except Exception:
            pass

    def stop(self):
        self.running = False

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.state)

camera = CameraWorker(source=CAMERA_SOURCE)

app = FastAPI(title="GreenMind Cam Backend", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

class Status(BaseModel):
    people_count: int
    brightness: float
    light_on: bool
    alert_ready: bool
    last_alert_ts: Optional[float]
    last_alert_image_url: Optional[str]
    last_alert_video_url: Optional[str]
    last_error: Optional[str]

@app.on_event("startup")
def on_startup():
    camera.start()

@app.on_event("shutdown")
def on_shutdown():
    camera.stop()

@app.get("/health")
def health():
    return {"ok": True, "firebase": firebase_ok, "twilio": twilio_ok}

@app.get("/status", response_model=Status)
def status():
    st = camera.get_state()
    now = time.time()
    empty_since = st.get("empty_since")
    alert_ready = False
    if st["light_on"] and st["people_count"] == 0 and empty_since:
        alert_ready = (now - empty_since) >= ALERT_EMPTY_SECONDS
    return Status(
        people_count=int(st["people_count"]),
        brightness=float(st["brightness"]),
        light_on=bool(st["light_on"]),
        alert_ready=bool(alert_ready),
        last_alert_ts=st["last_alert_ts"],
        last_alert_image_url=st.get("last_alert_image_url"),
        last_alert_video_url=st.get("last_alert_video_url"),
        last_error=st.get("last_error"),
    )

@app.get("/snapshot")
def snapshot():
    st = camera.get_state()
    frame = st.get("last_frame")
    if frame is None:
        return JSONResponse({"error": "No frame"}, status_code=503)
    ret, jpeg = cv2.imencode(".jpg", frame)
    if not ret:
        return JSONResponse({"error": "Encode failed"}, status_code=500)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

@app.get("/alerts")
def list_alerts(limit: int = 20):
    if firebase_ok and db is not None:
        try:
            docs = db.collection("alerts").order_by("ts", direction=firestore.Query.DESCENDING).limit(limit).stream()
            items: List[Dict[str, Any]] = []
            for d in docs:
                it = d.to_dict()
                it["id"] = d.id
                items.append(it)
            return {"items": items}
        except Exception as e:
            return {"items": [], "error": str(e)}
    else:
        return {"items": []}
