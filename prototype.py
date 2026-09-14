import cv2
import dlib
import math
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import threading
import pywhatkit as pwk
import pyautogui
import requests

# Q-learning parameters
alpha = 0.15
gamma = 0.92
epsilon = 0.95
epsilon_min = 0.05
epsilon_decay = 0.95

# Initial weights
weight_eor = 0.4
weight_blink_count = 0.4
weight_yawn = 0.1
weight_head_tilt = 0.1

# Initialize Q-table
q_table = {
    'eor': [weight_eor,weight_eor],
    'blink': [weight_blink_count,weight_blink_count],
    'yawn': [weight_yawn,weight_yawn],
    'head': [weight_head_tilt,weight_head_tilt]
}

# Lock to synchronize access to shared resources
shared_lock = threading.Lock()

# Variables for face detection and landmark prediction
face_data = {
    'faces': [],
    'landmarks': None,
    'shape': None,
    'gray': None
}

# Event to signal the face detection thread to stop
stop_face_detection_event = threading.Event()

# Sound-related variables
sound_lock = threading.Lock()
sound_thread_active = False
stop_sound_event = threading.Event()

pygame.mixer.init()

# Constants
CALIBRATION_FRAMES = 100
EMA_ALPHA = 0.2
YAWN_CALIBRATION_FRAMES = 100
YAWN_THRESH_FACTOR = 2.5
MIN_YAWN_THRESH = 10
MAX_YAWN_THRESH = 30

def get_my_ip_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()

        location_info = {
            "IP": data.get("ip"),
            "City": data.get("city"),
            "Region": data.get("region"),
            "Country": data.get("country"),
            "Location (lat,long)": data.get("loc"),
            "Org": data.get("org"),
            "Postal": data.get("postal"),
            "Timezone": data.get("timezone")
        }

        return location_info
    except Exception as e:
        return {"error": str(e)}

def send_whatsapp_location_instantly(phone_number):
    location = get_my_ip_location()

    if "error" in location:
        print("Error getting location:", location["error"])
        return

    # Prepare message
    message = (
        f"📍 IP Location Details:\n"
        f"IP: {location['IP']}\n"
        f"City: {location['City']}\n"
        f"Region: {location['Region']}\n"
        f"Country: {location['Country']}\n"
        f"Lat,Long: {location['Location (lat,long)']}\n"
        f"Org: {location['Org']}\n"
        f"Timezone: {location['Timezone']}"
    )

    # Send message instantly
    pwk.sendwhatmsg_instantly(phone_number, message, 10,True,4)
    time.sleep(2)
    pyautogui.press("enter")

def play_alert_sound(overall_score):
    global sound_lock, sound_thread_active, stop_sound_event
    with sound_lock:
        if overall_score > 0.8:
            pygame.mixer.music.load(r"C:\Users\ASUS\OneDrive\Desktop\project\alarm-1-with-reverberation-30031.mp3")
            pygame.mixer.music.play(loops=-1)
            send_whatsapp_location_instantly("+919836726369")
        elif overall_score > 0.4:
            pygame.mixer.music.load(r"C:\Users\ASUS\OneDrive\Desktop\project\alarm-1-with-reverberation-30031.mp3")
            pygame.mixer.music.play(loops=-1)

    stop_sound_event.wait(timeout=1)
    pygame.mixer.music.stop()
    sound_thread_active = False

def check_and_alert_sound(overall_score):
    global sound_thread_active, stop_sound_event
    if overall_score > 0.4 and not sound_thread_active:
        stop_sound_event.clear()
        threading.Thread(target=play_alert_sound, args=(overall_score,), daemon=True).start()
        sound_thread_active = True
    elif overall_score <= 0.4 and sound_thread_active:
        stop_sound_event.set()

def detect_faces_and_landmarks(detector, predictor):
    global face_data, shared_lock, stop_face_detection_event
    while not stop_face_detection_event.is_set():
        with shared_lock:
            gray = face_data['gray']
            if gray is not None:
                face_data['faces'] = detector(gray)
                if len(face_data['faces']) > 0:
                    face_data['landmarks'] = predictor(gray, face_data['faces'][0])
                    face_data['shape'] = face_utils.shape_to_np(face_data['landmarks'])
                else:
                    face_data['landmarks'] = None
                    face_data['shape'] = None
        time.sleep(0.1)

def cal_eor(eye_points):
    vertical_dist = dist.euclidean(eye_points[1], eye_points[5])
    horizontal_dist = dist.euclidean(eye_points[0], eye_points[3])
    return vertical_dist / horizontal_dist if horizontal_dist != 0 else 0

def cal_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    return dist.euclidean(np.mean(top_lip, axis=0), np.mean(low_lip, axis=0))

def cal_smile(shape):
    mouth = shape[48:68]
    left_corner = np.mean(mouth[:3], axis=0)
    right_corner = np.mean(mouth[6:9], axis=0)
    return dist.euclidean(left_corner, right_corner)

def update_q_table(state, action, reward):
    with shared_lock:
        if state not in q_table:
            q_table[state] = [weight_eor, weight_blink_count, weight_yawn, weight_head_tilt]
        if action < len(q_table[state]):
            max_future_q = max(q_table[state])
            q_table[state][action] += alpha * (reward + gamma * max_future_q - q_table[state][action])

def select_action(state):
    global epsilon
    with shared_lock:
        if state not in q_table:
            return 0
        return np.random.choice([0, 1]) if np.random.random() < epsilon else np.argmax(q_table[state])

def decay_epsilon():
    global epsilon
    with shared_lock:
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

def calculate_reward(state, action):
    if state == 'eor':
        return 2 if action == 1 else -1
    elif state == 'yawn':
        return 2 if action == 1 else -0.5
    elif state in ['blink', 'head']:
        return 1
    return 0

def dynamic_calibration(pitch, roll, yaw):
    global reference_pitch, reference_roll, reference_yaw, calibration_start_time
    if time.time() - calibration_start_time >= 20:
        reference_pitch = pitch
        reference_roll = roll
        reference_yaw = yaw
        calibration_start_time = time.time()

def estimate_face_distance(eye_distance_pixels):
    return (6.5 * 950) / eye_distance_pixels  # 6.5cm avg eye distance, 950px focal length

def adjust_focal_length(face_distance_cm, default_focal_length=950):
    if face_distance_cm < 30:
        return default_focal_length * 1.5
    elif face_distance_cm > 80:
        return default_focal_length * 0.8
    return default_focal_length

def main():
    global face_data, stop_face_detection_event, calibration_start_time, yawn_values, yawn_thresh

    # Initialize models and video
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\ASUS\OneDrive\Desktop\project\shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture('http://192.168.140.42:81/stream')
    
    # Setup display window
    cv2.namedWindow("Drowsiness Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drowsiness Detection", 1024, 768)
    scale_x, scale_y = 1024/640, 768/480

    # Start face detection thread
    face_detection_thread = threading.Thread(target=detect_faces_and_landmarks, 
                                          args=(detector, predictor), daemon=True)
    face_detection_thread.start()

    
    # Initialize variables
    calibration_frames = CALIBRATION_FRAMES
    eor_values = []
    eor_thresh = None
    blink_flag = False

    # Yawn threshold calibration
    yawn_values = []  # Stores lip distance values for calibration
    yawn_thresh = None  # Dynamic yawn threshold


    consecutive_frames = 0
    frames_to_confirm_blink = 7
    blink_count = 0
    prev_eor = 0

    ptime = time.time()
    yawn_detected = False
    eor_alert_start_time = None
    show_yawn_message = False
    freeze_blink_count = False
    freeze_blink_when_smiling = False
    blink_alert_threshold = 6
    blink_alert_duration = 20
    blink_alert_start_time = None
    blink_alert_display_time = None
    blink_alert_displayed = False
    point_eor = 0
    point_yawn = 0
    point_blink = 0
    point_head = 0
    reset_interval = 21
    overall_score_reset_start_time = time.time()

    pitch_threshold = 5
    roll_threshold = 20
    yaw_threshold = 8
    calibration_start_time = time.time()
    reference_pitch = None
    reference_roll = 0
    reference_yaw = None

    # Variables for smoothed head movement detection
    ema_pitch_diff = 0
    ema_roll_diff = 0
    ema_yaw_diff = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("[Warning] Stream lost. Trying to reconnect...")
            cap.release()
            time.sleep(3)  # Wait before retry
            for attempt in range(5):  # Try 5 times
                cap = cv2.VideoCapture('http://192.168.140.42:81/stream')
                time.sleep(2)  # Wait to stabilize stream
                ret, frame = cap.read()
                if ret and frame is not None and frame.size != 0:
                    print("[Info] Reconnected successfully.")
                    break
                else:
                    print(f"[Retry {attempt+1}/5] Reconnect failed.")
            else:
                print("[Error] Failed to reconnect after 5 attempts. Exiting.")
                break

        # Create upscaled display frame
        display_frame = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_CUBIC)
        
        # Process original resolution frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with shared_lock:
            face_data['gray'] = gray
            faces = face_data['faces']
            landmarks = face_data['landmarks']
            shape = face_data['shape']

        if len(faces) > 0 and landmarks is not None:
            # Calculate FPS
            ctime = time.time()
            fps = int(1/(ctime - ptime)) if ptime != ctime else 0
            ptime = ctime
            
            # Draw on upscaled frame (scale all coordinates)
            def sp(x, y): return (int(x*scale_x), int(y*scale_y))
            
            cv2.putText(display_frame, f'FPS: {fps}', (20, display_frame.shape[0]-20), 
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (100, 200, 0), 2)

            # Head movement calculation (on original frame)
            nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            roll = math.degrees(math.atan2(delta_y, delta_x))
            
            # Head pose estimation
            eye_distance_pixels = dist.euclidean(left_eye, right_eye)
            face_distance_cm = estimate_face_distance(eye_distance_pixels)
            focal_length = adjust_focal_length(face_distance_cm)
            center_x, center_y = frame.shape[1]/2, frame.shape[0]/2
            scale = focal_length/frame.shape[1]
            yaw = math.degrees(math.atan2((nose[0]-center_x)*scale, focal_length))
            pitch = math.degrees(math.atan2((nose[1]-center_y)*scale, focal_length))

            # Dynamic calibration
            if reference_yaw is None: reference_yaw = yaw
            if reference_pitch is None: reference_pitch = pitch
            if reference_roll is None: reference_roll = roll
            dynamic_calibration(pitch, roll, yaw)

            # Calculate differences
            pitch_diff = pitch - reference_pitch
            roll_diff = roll - reference_roll
            yaw_diff = yaw - reference_yaw
            
            # Apply EMA smoothing
            ema_pitch_diff = EMA_ALPHA*pitch_diff + (1-EMA_ALPHA)*ema_pitch_diff
            ema_roll_diff = EMA_ALPHA*roll_diff + (1-EMA_ALPHA)*ema_roll_diff
            ema_yaw_diff = EMA_ALPHA*yaw_diff + (1-EMA_ALPHA)*ema_yaw_diff
            
            cv2.putText(display_frame, f"Pitch: {ema_pitch_diff:.2f} Roll: {ema_roll_diff:.2f} Yaw: {ema_yaw_diff:.2f}", 
                       sp(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Yawn detection
            lip_dist = cal_yawn(shape)

            # Yawn threshold calibration
            if len(yawn_values) < YAWN_CALIBRATION_FRAMES:
                yawn_values.append(lip_dist)
                cv2.putText(display_frame, f"Calibrating Yawn... {len(yawn_values)}/{YAWN_CALIBRATION_FRAMES}", sp(20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif yawn_thresh is None:
                avg_yawn = np.mean(yawn_values)
                yawn_thresh = avg_yawn * YAWN_THRESH_FACTOR
                yawn_thresh = max(yaw_threshold, min(yawn_thresh, MAX_YAWN_THRESH))  # Clamp within bounds
                print(f"Yawn Calibration Complete. Dynamic Yawn Threshold: {yawn_thresh}")
            else:
                pass
            
            if yawn_thresh is not None and lip_dist > yawn_thresh:
                if not yawn_detected:
                    yawn_detected = True
                    show_yawn_message = True
                    freeze_blink_count = True

            else:
                yawn_detected = False
                show_yawn_message = False
                freeze_blink_count = False

            if show_yawn_message:
                cv2.putText(display_frame, "Yawning Detected", sp(20, frame.shape[0]-180), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

            # Eye detection
            left_eye, right_eye = shape[42:48], shape[36:42]
            avg_eor = (cal_eor(left_eye) + cal_eor(right_eye))/2

            # EOR calibration
            if len(eor_values) < calibration_frames:
                eor_values.append(avg_eor)
                cv2.putText(display_frame, f"Calibrating... {len(eor_values)}/{calibration_frames}", 
                           sp(20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif eor_thresh is None:
                eor_thresh = np.mean(eor_values)*0.76
                print(f"Calibration complete. Dynamic EOR Threshold: {eor_thresh}")
            else:
                eor_thresh = np.mean(eor_values[-calibration_frames:])*0.76

            cv2.putText(display_frame, f'EOR: {avg_eor:.2f}', (20, display_frame.shape[0]-50), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f'Blink Count: {blink_count}', (20, display_frame.shape[0]-80), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

            # Drowsiness detection based on EOR
            if eor_thresh is not None and avg_eor < eor_thresh:
                if eor_alert_start_time is None:
                    eor_alert_start_time = time.time()
                else:
                    if time.time() - eor_alert_start_time >= 2 and not show_yawn_message:
                        cv2.putText(display_frame, "Drowsiness Alert!", (20, display_frame.shape[0] - 210), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
                        point_eor = 3
                        action = select_action('eor')
                        reward = calculate_reward('eor', action)
                        update_q_table('eor', action, reward)
                        if (abs(ema_pitch_diff) > pitch_threshold or
                            abs(ema_roll_diff) > roll_threshold or
                            abs(ema_yaw_diff) > yaw_threshold):
                            cv2.putText(display_frame, "Alert!!! Wake Up...", sp(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            point_head = 3
                            action = select_action('head')
                            reward = calculate_reward('head', action)
                            update_q_table('head', action, reward)
                    elif time.time() - eor_alert_start_time < 1 and not show_yawn_message:
                        point_eor = 0
                    elif time.time() - eor_alert_start_time >= 2 and show_yawn_message:
                        cv2.putText(display_frame, "Drowsiness Alert!", (20, display_frame.shape[0] - 210), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
                        point_yawn = 3
                        action = select_action('yawn')
                        reward = calculate_reward('yawn', action)
                        update_q_table('yawn', action, reward)
                        if (abs(ema_pitch_diff) > pitch_threshold or
                            abs(ema_roll_diff) > roll_threshold or
                            abs(ema_yaw_diff) > yaw_threshold):
                            cv2.putText(display_frame, "Alert!!! Wake Up...", sp(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            point_head = 3
                            action = select_action('head')
                            reward = calculate_reward('head', action)
                            update_q_table('head', action, reward)
                    elif time.time() - eor_alert_start_time < 2 and show_yawn_message:
                        point_yawn = 2
                        action = select_action('yawn')
                        reward = calculate_reward('yawn', action)
                        update_q_table('yawn', action, reward)
                        
            else:
                eor_alert_start_time = None
            # Calculate EOR

            if prev_eor == 0:
                prev_eor = avg_eor
            else:
                if not freeze_blink_count and not freeze_blink_when_smiling:
                    if eor_thresh is not None and avg_eor < eor_thresh and not blink_flag:
                        consecutive_frames += 1
                        if consecutive_frames >= frames_to_confirm_blink:
                            blink_count += 1
                            blink_flag = True
                    elif eor_thresh is not None and avg_eor > eor_thresh and blink_flag:
                        blink_flag = False
                        consecutive_frames = 0
            prev_eor = avg_eor

            # Smile detection
            smile_thresh = 60
            smile_distance = cal_smile(shape)
            if smile_distance > smile_thresh:
                freeze_blink_when_smiling = True
            else:
                freeze_blink_when_smiling = False

            # Blink count alert
            if blink_count < blink_alert_threshold:
                if blink_alert_start_time is None:
                    blink_alert_start_time = time.time()
                else:
                    alert_duration = time.time() - blink_alert_start_time
                    if alert_duration >= blink_alert_duration and not blink_alert_displayed:
                        cv2.putText(display_frame, "Blink Count Alert! Take a break.", (20, display_frame.shape[0] - 240),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
                        point_blink = 3
                        action = select_action('blink')
                        reward = calculate_reward('blink', action)  # Use the refactored function
                        update_q_table('blink', action, reward)
                        if blink_alert_display_time is None:
                            blink_alert_display_time = time.time()
                        elif time.time() - blink_alert_display_time >= 2:
                            blink_count = 0
                            blink_alert_start_time = None
                            blink_alert_display_time = None
                            blink_alert_displayed = True
                    else:
                        blink_alert_displayed = False
            else:
                if blink_alert_start_time is not None and time.time() - blink_alert_start_time >= blink_alert_duration:
                    point_blink = 0
                    blink_count = 0
                    blink_alert_start_time = None
                    blink_alert_display_time = None

            # Reset overall score periodically
            if time.time() - overall_score_reset_start_time >= reset_interval:
                overall_score_reset_start_time = time.time()
                overall_score = 0
                point_eor = 0
                point_yawn = 0
                point_blink = 0
                point_head = 0

            # Calculate overall score
            overall_score = (q_table['eor'][1] * point_eor + q_table['blink'][1] * point_blink +
                             q_table['yawn'][1] * point_yawn + q_table['head'][1] * point_head) / 12

            # Normalize the score between 0 and 1
            max_possible_score = 3  # Since each point_* can be at most 3
            normalized_score = overall_score / max_possible_score
            normalized_score = max(0, min(1, normalized_score))

            if normalized_score > 0.8:
                drowsiness_level = "High"
            elif normalized_score > 0.4:
                drowsiness_level = "Moderate"
            else:
                drowsiness_level = "Low"

            cv2.putText(display_frame, f'Score: {normalized_score:.2f}', (20, display_frame.shape[0] - 110), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f'Drowsiness Level: {drowsiness_level}', (20, display_frame.shape[0] - 140), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

            decay_epsilon()

            check_and_alert_sound(normalized_score)


        cv2.imshow("Drowsiness Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_face_detection_event.set()
    face_detection_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()