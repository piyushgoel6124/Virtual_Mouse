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
        rx = 0.05 * frame_width
        ry = 0.05 * frame_height
        adjusted_frame_width = 0.8 * frame_width
        adjusted_frame_height = 0.8 * frame_height

        # Draw ROI rectangle
        cv2.rectangle(frame, (int(rx), int(ry)), (int(adjusted_frame_width), int(adjusted_frame_height)), (0, 255, 0), 2)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand_landmarks_1, hand_landmarks_2 = results.multi_hand_landmarks

            # Extract coordinates for both hands
            def extract_landmarks(hand_landmarks):
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                return thumb, index, middle

            thumb_1, index_1, middle_1 = extract_landmarks(hand_landmarks_1)
            thumb_2, index_2, middle_2 = extract_landmarks(hand_landmarks_2)

            # Convert normalized coordinates to pixel coordinates
            def to_pixel_coords(landmark, frame_width, frame_height):
                return int(landmark.x * frame_width), int(landmark.y * frame_height)

            thumb_1_x, thumb_1_y = to_pixel_coords(thumb_1, adjusted_frame_width, adjusted_frame_height)
            thumb_2_x, thumb_2_y = to_pixel_coords(thumb_2, adjusted_frame_width, adjusted_frame_height)
            middle_1_x, middle_1_y = to_pixel_coords(middle_1, adjusted_frame_width, adjusted_frame_height)
            middle_2_x, middle_2_y = to_pixel_coords(middle_2, adjusted_frame_width, adjusted_frame_height)
            index_1_x, index_1_y = to_pixel_coords(index_1, adjusted_frame_width, adjusted_frame_height)
            index_2_x, index_2_y = to_pixel_coords(index_2, adjusted_frame_width, adjusted_frame_height)

            # Calculate distances
            def calculate_distance(x1, y1, x2, y2):
                return math.hypot(x2 - x1, y2 - y1)

            index_distance = calculate_distance(index_1_x, index_1_y, index_2_x, index_2_y)
            thumb_distance = calculate_distance(thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y)
            thumb_middle_distance_1 = calculate_distance(thumb_1_x, thumb_1_y, middle_2_x, middle_2_y)
            thumb_middle_distance_2 = calculate_distance(thumb_2_x, thumb_2_y, middle_1_x, middle_1_y)

            # Move the cursor based on index fingers
            if index_distance < 20:
                cursor_x = (index_1_x + index_2_x) / 2
                cursor_y = (index_1_y + index_2_y) / 2
                pyautogui.moveTo(cursor_x * (screen_width / adjusted_frame_width), cursor_y * (screen_height / adjusted_frame_height))

            # Left-click with thumbs
            if thumb_distance < 30:
                pyautogui.click()
                pyautogui.sleep(0.5)

            # Right-click with middle fingers
            if calculate_distance(middle_1_x, middle_1_y, middle_2_x, middle_2_y) < 30:
                pyautogui.rightClick()
                pyautogui.sleep(0.5)

            # Mouse hold with thumb and middle combinations
            if thumb_middle_distance_1 < 30 or thumb_middle_distance_2 < 30:
                if not mouse_held:
                    pyautogui.mouseDown()
                    mouse_held = True
                else:
                    if thumb_middle_distance_1 > 30 and thumb_middle_distance_2 > 30:
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
