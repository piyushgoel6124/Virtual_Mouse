import mediapipe as mp
import cv2
import pyautogui
import math
import keyboard

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Initialize OpenCV
cap = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False
paused = False
mouse_held = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Check if Ctrl+Alt+/ is pressed
    if keyboard.is_pressed('ctrl+alt+/'):
        paused = not paused
        keyboard.wait('ctrl+alt+/')  # Wait until the keys are released

    if not paused:
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Get the dimensions of the frame
        frame_height, frame_width, _ = frame.shape
        rx = 0.05 * frame_height
        ry = 0.05 * frame_width
        frame_height = 0.8 * frame_height
        frame_width = 0.8 * frame_width

        cv2.rectangle(frame, (int(rx), int(ry)), (int(frame_width), int(frame_height)), (0, 255, 0), 2)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand_landmarks_1, hand_landmarks_2 = results.multi_hand_landmarks

            # Extract coordinates of thumb, index, and middle fingers for both hands
            thumb_1 = hand_landmarks_1.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_1 = hand_landmarks_1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_1 = hand_landmarks_1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            thumb_2 = hand_landmarks_2.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_2 = hand_landmarks_2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_2 = hand_landmarks_2.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized coordinates to pixel coordinates
            index_1_x, index_1_y = int(index_1.x * frame_width), int(index_1.y * frame_height)
            index_2_x, index_2_y = int(index_2.x * frame_width), int(index_2.y * frame_height)

            thumb_1_x, thumb_1_y = int(thumb_1.x * frame_width), int(thumb_1.y * frame_height)
            thumb_2_x, thumb_2_y = int(thumb_2.x * frame_width), int(thumb_2.y * frame_height)

            middle_1_x, middle_1_y = int(middle_1.x * frame_width), int(middle_1.y * frame_height)
            middle_2_x, middle_2_y = int(middle_2.x * frame_width), int(middle_2.y * frame_height)

            # Calculate distances between corresponding fingers of both hands
            index_distance = math.hypot(index_2_x - index_1_x, index_2_y - index_1_y)
            thumb_distance = math.hypot(thumb_2_x - thumb_1_x, thumb_2_y - thumb_1_y)
            middle_distance = math.hypot(middle_2_x - middle_1_x, middle_2_y - middle_1_y)
            thumb_middle_distance = math.hypot(thumb_1_x - middle_2_x, thumb_1_y - middle_2_y)

            # Move the cursor
            if index_distance < 30:
                cursor_x = (index_1_x + index_2_x) / 2
                cursor_y = (index_1_y + index_2_y) / 2
                pyautogui.moveTo(cursor_x * (screen_width / frame_width), cursor_y * (screen_height / frame_height))

            # Left-click
            if thumb_distance < 30:
                pyautogui.click()
                pyautogui.sleep(0.5)

            # Right-click
            if middle_distance < 30:
                pyautogui.rightClick()
                pyautogui.sleep(0.5)

            # Mouse down and up for selection
            if thumb_middle_distance < 30:
                if not mouse_held:
                    pyautogui.mouseDown()
                    mouse_held = True
                else:
                    pyautogui.mouseUp()
                    mouse_held = False

    # Display the frame in fullscreen mode
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
