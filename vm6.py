import cv2
import mediapipe as mp
import pyautogui

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize variables
mouse_held = False

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand landmarks
    results = hands.process(rgb_frame)
    cursor_x, cursor_y = None, None

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks
        if len(landmarks) == 2:
            hand_1, hand_2 = landmarks[0], landmarks[1]

            # Calculate distances for both hands
            index_1 = hand_1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_1 = hand_1.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_1 = hand_1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            index_2 = hand_2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_2 = hand_2.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_2 = hand_2.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Extract coordinates and scale them
            index_1_x, index_1_y = index_1.x * frame_width, index_1.y * frame_height
            thumb_1_x, thumb_1_y = thumb_1.x * frame_width, thumb_1.y * frame_height
            middle_1_x, middle_1_y = middle_1.x * frame_width, middle_1.y * frame_height

            index_2_x, index_2_y = index_2.x * frame_width, index_2.y * frame_height
            thumb_2_x, thumb_2_y = thumb_2.x * frame_width, thumb_2.y * frame_height
            middle_2_x, middle_2_y = middle_2.x * frame_width, middle_2.y * frame_height

            # Calculate distances
            index_distance = ((index_1_x - index_2_x) ** 2 + (index_1_y - index_2_y) ** 2) ** 0.5
            thumb_distance = ((thumb_1_x - thumb_2_x) ** 2 + (thumb_1_y - thumb_2_y) ** 2) ** 0.5
            thumb_middle_distance_1 = ((thumb_1_x - middle_1_x) ** 2 + (thumb_1_y - middle_1_y) ** 2) ** 0.5
            thumb_middle_distance_2 = ((thumb_2_x - middle_2_x) ** 2 + (thumb_2_y - middle_2_y) ** 2) ** 0.5

            # Cursor movement when index fingers are close
            if index_distance < 20:
                cursor_x = (index_1_x + index_2_x) / 2
                cursor_y = (index_1_y + index_2_y) / 2
                transformed_x = ((4 / 3) * cursor_x - (100 / 3)) * (screen_width / frame_width)
                transformed_y = (cursor_y - 50) * (screen_height / frame_height)
                pyautogui.moveTo(transformed_x, transformed_y)

            # Clicking action when thumbs are close
            if thumb_distance < 30:
                pyautogui.click()
                pyautogui.sleep(0.5)

            # Mouse hold and release based on thumb-to-middle distances
            if thumb_middle_distance_1 < 30 or thumb_middle_distance_2 < 30:
                if not mouse_held:
                    pyautogui.mouseDown()
                    mouse_held = True
                transformed_x = ((4 / 3) * middle_1_x - (100 / 3)) * (screen_width / frame_width)
                transformed_y = (middle_1_y - 50) * (screen_height / frame_height)
                pyautogui.moveTo(transformed_x, transformed_y)
            elif mouse_held and thumb_middle_distance_1 > 30 and thumb_middle_distance_2 > 30:
                pyautogui.mouseUp()
                mouse_held = False

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
