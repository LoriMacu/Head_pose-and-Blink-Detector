import numpy as np
import cv2
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Webcam Resolution: {frame_width}x{frame_height}")

# Known constant: average horizontal iris diameter in mm
iris_diameter_mm = 11.7

# Blink detection parameters
EAR_THRESHOLD = 0.16
BLINK_FRAMES = 1  # Minimum frames to consider a blink

# Blink counters
left_blink_counter = 0
right_blink_counter = 0
total_left_blinks = 0
total_right_blinks = 0

# EAR calculation function
def calculate_ear(landmarks, indices, frame_width, frame_height):
    # Extract coordinates of the eye landmarks
    eye = np.array([(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in indices])
    # Compute the distances between the vertical eye landmarks
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    # Compute the distance between the horizontal eye landmarks
    horizontal = np.linalg.norm(eye[0] - eye[3])
    # EAR calculation
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect up to 5 faces
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True  # Enables iris landmarks
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
        if not ret:
            break

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Draw face mesh for this face
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

                # Draw iris connections for this face
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

                # Extract eye positions
                left_eye_center = face_landmarks.landmark[468]  # Left iris center
                right_eye_center = face_landmarks.landmark[473]  # Right iris center

                left_eye = (int(left_eye_center.x * frame.shape[1]), int(left_eye_center.y * frame.shape[0]))
                right_eye = (int(right_eye_center.x * frame.shape[1]), int(right_eye_center.y * frame.shape[0]))

                # Calculate iris diameter in pixels for depth estimation
                left_eye_inner = face_landmarks.landmark[469]  # Inner left iris
                left_eye_outer = face_landmarks.landmark[471]  # Outer left iris

                iris_diameter_px = np.linalg.norm([
                    (left_eye_outer.x - left_eye_inner.x) * frame.shape[1],
                    (left_eye_outer.y - left_eye_inner.y) * frame.shape[0],
                ])

                # Estimate depth using the pinhole camera model
                focal_length_px = frame_width  # Assuming focal length in pixels matches frame width
                depth_mm = (iris_diameter_mm * focal_length_px) / iris_diameter_px
                depth_cm = depth_mm / 10  # Convert to cm

                # Head pose estimation
                face_3d = []
                face_2d = []
                for landmark_idx, lm in enumerate(face_landmarks.landmark):
                    if landmark_idx in {1, 33, 263, 61, 291, 199}:  # Key points for pose estimation
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                cam_matrix = np.array([[focal_length_px, 0, frame.shape[1] / 2],
                                       [0, focal_length_px, frame.shape[0] / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                z_angle = angles[2] * 360

                head_pose = "Forward"
                if y_angle < -10:
                    head_pose = "Looking Left"
                elif y_angle > 10:
                    head_pose = "Looking Right"
                elif x_angle < -10:
                    head_pose = "Looking Down"
                elif x_angle > 10:
                    head_pose = "Looking Up"
                #elif -0.05 < z_angle < 0.05:
                 #   head_pose = "Roll"

                # Blink detection logic
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                # Calculate EAR for left and right eyes
                left_ear = calculate_ear(face_landmarks.landmark, left_eye_indices, frame_width, frame_height)
                right_ear = calculate_ear(face_landmarks.landmark, right_eye_indices, frame_width, frame_height)

                if left_ear < EAR_THRESHOLD:
                    left_blink_counter += 1
                else:
                    if left_blink_counter >= BLINK_FRAMES:
                        total_left_blinks += 1
                    left_blink_counter = 0

                if right_ear < EAR_THRESHOLD:
                    right_blink_counter += 1
                else:
                    if right_blink_counter >= BLINK_FRAMES:
                        total_right_blinks += 1
                    right_blink_counter = 0

                vertical_offset = 30  # Initial vertical offset for text
                line_spacing = 20     # Spacing between lines
                # Display EAR values and blink count
                #cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                #cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Left Blinks: {total_left_blinks}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Blinks: {total_right_blinks}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Eye: {left_eye}", (10, vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                vertical_offset += line_spacing  # Move to the next line
                cv2.putText(frame, f"Right Eye: {right_eye}", (10, vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                vertical_offset += line_spacing  # Move to the next line
                # Display depth, head pose, and eye positions
                vertical_offset = 120  # Offset for additional text
                cv2.putText(frame, f"Face {idx + 1} - Depth: {depth_cm:.2f} cm", (10, vertical_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                vertical_offset += 20
                cv2.putText(frame, f"Head Pose: {head_pose}", (10, vertical_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                vertical_offset += 20
                cv2.putText(frame, f"Angles (deg) X={x_angle:.1f}, Y={y_angle:.1f}, Z={z_angle:.1f}",
                            (10, vertical_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                 # Display the frame
        cv2.imshow("Multi-Face Eye, Head Pose, Depth, and Blink Detection", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
