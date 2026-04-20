import cv2
import math
import mediapipe as mp
import numpy as np
from collections import deque
from dearpygui.dearpygui import *

# ---------------- Mediapipe Setup ----------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageFormat = mp.ImageFormat
Image = mp.Image

# ---------------- Landmark IDs ----------------
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_EAR = 7
LEFT_HIP = 23

# ---------------- Utilities ----------------
def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y1 - y2
    return abs(math.degrees(math.atan2(dx, dy)))

# ---------------- Analyzer ----------------
class PostureAnalyzer:
    def __init__(self, model_path):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.colors = {"good": (0,170,0), "warn": (0,190,255), "bad": (0,0,200)}
        self.neck_hist = deque(maxlen=20)
        self.torso_hist = deque(maxlen=20)
        self.head_hist = deque(maxlen=20)

    def process(self, frame, timestamp, thresholds):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp)

        if not result.pose_landmarks:
            return frame, 0, 0, 0, "No Detection", [], 0

        lm = result.pose_landmarks[0]
        sh = (int((lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x)/2*w),
              int((lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y)/2*h))
        l_sh = (int(lm[LEFT_SHOULDER].x*w), int(lm[LEFT_SHOULDER].y*h))
        r_sh = (int(lm[RIGHT_SHOULDER].x*w), int(lm[RIGHT_SHOULDER].y*h))
        ear = (int(lm[LEFT_EAR].x*w), int(lm[LEFT_EAR].y*h))
        hip = (int(lm[LEFT_HIP].x*w), int(lm[LEFT_HIP].y*h))

        # Metrics
        neck = calculate_angle(*sh, *ear)
        torso = calculate_angle(*hip, *sh)
        head_forward = abs(ear[0] - sh[0])

        # Smooth values
        self.neck_hist.append(neck)
        self.torso_hist.append(torso)
        self.head_hist.append(head_forward)
        neck = sum(self.neck_hist)/len(self.neck_hist)
        torso = sum(self.torso_hist)/len(self.torso_hist)
        head_forward = sum(self.head_hist)/len(self.head_hist)

        # Penalties & score
        neck_pen = max(0, neck - thresholds["neck"])
        torso_pen = max(0, torso - thresholds["torso"])
        head_pen = max(0, head_forward - thresholds["head"])
        penalty = neck_pen*0.35 + torso_pen*0.45 + head_pen*0.2
        score = max(0, 100-int(penalty*2))
        if torso>30: score=min(score,50)

        # Status
        if score>80: status="ALIGNED"; color=self.colors["good"]
        elif score>60: status="SLIGHTLY OFF"; color=self.colors["warn"]
        else: status="MISALIGNED"; color=self.colors["bad"]

        # Draw skeleton
        frame = self.draw_skeleton(frame,l_sh,r_sh,ear,hip,color)
        cv2.putText(frame,f"{status} ({score})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        return frame, neck, torso, head_forward, status, [l_sh,r_sh,ear,hip], score

    def draw_skeleton(self, frame, l_sh,r_sh,ear,hip,color):
        cv2.line(frame,l_sh,r_sh,color,2)
        cv2.line(frame,ear,l_sh,color,2)
        cv2.line(frame,hip,l_sh,color,2)
        for p in [l_sh,r_sh,ear,hip]:
            cv2.circle(frame,p,5,color,-1)
        return frame

# ---------------- App ----------------
class PostureApp:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)//2*2)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//2*2)

        self.analyzer = PostureAnalyzer(model_path)
        self.output_video = cv2.VideoWriter(
            "out.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        self.thresholds = {"neck":20,"torso":10,"head":35}
        self.neck_hist,self.torso_hist,self.head_hist,self.score_hist = [],[],[],[]

        self.setup_gui()

    def setup_gui(self):
        with window(label="Posture Analyzer", width=self.width+300, height=self.height+200):
            add_text("Thresholds")
            add_slider_int("Neck Angle",20,10,60,callback=self.update_threshold,user_data="neck")
            add_slider_int("Torso Angle",10,5,40,callback=self.update_threshold,user_data="torso")
            add_slider_int("Head Forward",35,10,60,callback=self.update_threshold,user_data="head")
            add_spacing(count=2)
            add_text("Live Video")
            self.video_widget = add_drawing("Video Feed", width=self.width, height=self.height)
            self.texture_tag = "frame_tex"
            add_raw_texture(self.texture_tag,self.width,self.height,np.zeros((self.height,self.width,3),dtype=np.uint8))
            draw_image(self.video_widget,(0,0),(self.width,self.height),self.texture_tag)
            add_spacing(count=1)
            add_text("Posture Metrics")
            self.score_text = add_text("Score: 0 | Status: --")
            self.neck_plot = add_plot("Neck Angle Trend",height=100)
            self.torso_plot = add_plot("Torso Angle Trend",height=100)
            self.head_plot = add_plot("Head Forward Trend",height=100)

    def update_threshold(self, sender, app_data, user_data):
        self.thresholds[user_data] = app_data

    def update(self, sender=None, app_data=None):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.output_video.release()
            stop_dearpygui()
            print("Video processing finished. Output saved to out.mp4")
            return

        timestamp = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        frame, neck, torso, head_forward, status, joints, score = self.analyzer.process(frame, timestamp, self.thresholds)

        # Resize & save video
        frame_bgr = cv2.resize(frame, (self.width, self.height))
        self.output_video.write(frame_bgr)

        # GUI display (RGB, no flip)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        set_raw_texture(self.texture_tag, frame_rgb)

        # Update metrics
        set_value(self.score_text, f"Score: {score} | Status: {status}")

        # Update histories
        self.neck_hist.append(neck)
        self.torso_hist.append(torso)
        self.head_hist.append(head_forward)
        self.score_hist.append(score)

        # Update plots
        clear_plot(self.neck_plot)
        add_line_series(self.neck_plot, "Neck", list(range(len(self.neck_hist))), list(self.neck_hist))
        clear_plot(self.torso_plot)
        add_line_series(self.torso_plot, "Torso", list(range(len(self.torso_hist))), list(self.torso_hist))
        clear_plot(self.head_plot)
        add_line_series(self.head_plot, "Head Forward", list(range(len(self.head_hist))), list(self.head_hist))

        # Schedule next frame
        set_frame_callback(self.update, int(1000 / self.fps))

    def run(self):
        set_frame_callback(self.update, int(1000 / self.fps))
        start_dearpygui()

# ---------------- Main ----------------
if __name__ == "__main__":
    video_path = "input.mp4"
    model_path = "pose_landmarker_lite.task"
    app = PostureApp(video_path, model_path)
    app.run()