import cv2
import os
import time
import pyautogui
import pywhatkit as kit
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
from threading import Lock, Thread
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
from queue import Queue

app = Flask(__name__)

model = YOLO("C:/Users/prave/OneDrive/Desktop/weapon_detection/best2.pt")

camera = None
camera_running = False
lock = Lock()

SAVE_FOLDER = "detected_frames"
os.makedirs(SAVE_FOLDER, exist_ok=True)

PHONE_NUMBER = "+916301946952"

tracker = DeepSort(max_age=15)

detected_objects = {}  # Dictionary to store saved frames for each track ID
color_map = {}
object_classes = {}  # Store class name for each tracked object ID

message_queue = Queue()


def get_random_color():
    return tuple(random.randint(100, 255) for _ in range(3))


def send_whatsapp_messages():
    while True:
        image_path, weapon_count, detected_classes = message_queue.get()
        if image_path:
            caption = f"🚨 Weapon Detected!\n🛑 Count: {weapon_count}\n🔹 Classes: {', '.join(detected_classes)}"
            time.sleep(2)
            kit.sendwhats_image(phone_no=PHONE_NUMBER, img_path=image_path, caption=caption)
            print("Waiting for WhatsApp Web to load...")
            time.sleep(3)
            pyautogui.press('enter')
            print(f"WhatsApp message sent with image: {image_path}")
        message_queue.task_done()

Thread(target=send_whatsapp_messages, daemon=True).start()


def generate_frames():
    global camera, camera_running, detected_objects, object_classes
    while True:
        with lock:
            if not camera_running or camera is None:
                break
            success, frame = camera.read()
            if not success:
                break

            results = model(frame, conf=0.5)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    if conf > 0.5:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

            tracked_objects = tracker.update_tracks(detections, frame=frame)

            for obj in tracked_objects:
                if not obj.is_confirmed():
                    continue

                track_id = obj.track_id
                ltrb = obj.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                if track_id not in object_classes:
                    for det in detections:
                        if abs(x1 - det[0][0]) < 10 and abs(y1 - det[0][1]) < 10:
                            object_classes[track_id] = model.names[det[2]]
                            break

                class_name = object_classes.get(track_id, None)
                if class_name is None:
                    continue

                if track_id not in color_map:
                    color_map[track_id] = get_random_color()
                color = color_map[track_id]

                if track_id not in detected_objects:
                    detected_objects[track_id] = True
                    timestamp = int(time.time())
                    image_path = os.path.join(SAVE_FOLDER, f"{class_name}_ID{track_id}_{timestamp}.jpg")
                    
                    frame_copy = frame.copy()
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_copy, f"{class_name} ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imwrite(image_path, frame_copy)
                    message_queue.put((image_path, 1, [class_name]))
                    print(f"Frame saved: {image_path}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('ind.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['GET'])
def start_camera():
    global camera, camera_running, detected_objects, object_classes
    with lock:
        if not camera_running:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({"status": "error", "message": "Failed to access camera"}), 500
            camera_running = True
            detected_objects = {}
            object_classes = {}
            return jsonify({"status": "started"})
    return jsonify({"status": "already running"})


@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    global camera, camera_running
    with lock:
        if camera_running:
            camera_running = False
            if camera:
                camera.release()
                camera = None
            return jsonify({"status": "stopped"})
    return jsonify({"status": "already stopped"})


@app.route('/get_saved_image', methods=['GET'])
def get_saved_image():
    if detected_objects:
        return jsonify({"message": "Images saved for detected objects."})
    return jsonify({"message": "No image detected yet"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
