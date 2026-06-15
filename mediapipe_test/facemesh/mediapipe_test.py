import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 1,
    smooth_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as pose:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        frame.flags.writeable = False
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        frame.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            h, w, _ = image.shape
            shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]. x * w),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)]
            
            elbow = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)]
            
            wrist = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]. x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)]
            
            right_arm_angle = calculate_angle(shoulder, elbow, wrist)
            print(f"Right Arm Angle: {right_arm_angle:.2f} deg")

            cv2.putText(image, f"{int(right_arm_angle)} deg",
                        tuple(elbow),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow('MediaPipe Pose Angle Counter', image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()