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
mouse_held = False  # To track mouse hold state

def calculate_distance(point1, point2):
    return math.hypot(point2.x - point1.x, point2.y - point1.y)
# Function to map hand positions to screen coordinates
def map_to_screen(hand_position, screen_width, screen_height):
    # Normalize hand position between 0 and 1
    normalized_x = hand_position.x
    normalized_y = hand_position.y

    # Apply desired mapping (scale the position based on the 0.2 and 0.8 positions)
    mapped_x = int(screen_width * (normalized_x - 0.2) / 0.6)
    mapped_y = int(screen_height * (normalized_y - 0.2) / 0.6)

    # Ensure that the mapped coordinates are within the screen bounds
    mapped_x = min(max(mapped_x, 0), screen_width)
    mapped_y = min(max(mapped_y, 0), screen_height)

    return mapped_x, mapped_y

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

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand1, hand2 = results.multi_hand_landmarks

            # Get relevant landmarks
            hand1_index = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand1_thumb = hand1.landmark[mp_hands.HandLandmark.THUMB_TIP]
            hand1_middle = hand1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            hand2_index = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand2_thumb = hand2.landmark[mp_hands.HandLandmark.THUMB_TIP]
            hand2_middle = hand2.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate distances
            index_to_index = calculate_distance(hand1_index, hand2_index)
            thumb_to_thumb = calculate_distance(hand1_thumb, hand2_thumb)
            middle_to_middle = calculate_distance(hand1_middle, hand2_middle)
            # Calculate distances for both thumb-middle combinations
            thumb_to_middle_1 = calculate_distance(hand1_thumb, hand2_middle)
            thumb_to_middle_2 = calculate_distance(hand1_middle, hand2_thumb)

            # Use the smallest distance to trigger mouse drag
            thumb_to_middle = min(thumb_to_middle_1, thumb_to_middle_2)

            # Move the cursor when index fingers are joined
            if index_to_index < 0.05 and middle_to_middle < 0.05:
                cursor_x, cursor_y = map_to_screen(hand1_index, screen_width, screen_height)
                pyautogui.moveTo(cursor_x, cursor_y)

            # Left click when thumbs are joined
            if thumb_to_thumb < 0.05:
                pyautogui.click()
                pyautogui.sleep(1)

            # Right click when middle fingers are joined
            if index_to_index < 0.05 and middle_to_middle > 0.05:
                pyautogui.rightClick()
                pyautogui.sleep(1)

            # Mouse drag with thumb + middle
            if thumb_to_middle < 0.05:
                if not mouse_held:
                    pyautogui.mouseDown()
                    mouse_held = True
            elif mouse_held:
                pyautogui.mouseUp()
                mouse_held = False

    # Display the frame
    # cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
