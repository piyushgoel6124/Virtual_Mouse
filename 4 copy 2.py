import mediapipe as mp
import cv2
import pyautogui
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Initialize OpenCV
cap = cv2.VideoCapture(0)
pyautogui.FAILSAFE=False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape
    rx=0.05*frame_height
    ry=0.05*frame_width
    frame_height = 0.8*frame_height
    frame_width= 0.8*frame_width


    cv2.rectangle(frame, (int(rx), int(ry)), (int(frame_width), int(frame_height )), (0, 255, 0), 2)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract coordinates of thumb, index, and middle fingers
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized coordinates to pixel coordinates
            thumb_x, thumb_y = int(thumb.x * frame_width), int(thumb.y * frame_height)
            index_x, index_y = int(index.x * frame_width), int(index.y * frame_height)
            middle_x, middle_y = int(middle.x * frame_width), int(middle.y * frame_height)

            # Draw red lines between fingers
            cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (0, 0, 255), 2)
            cv2.line(frame, (middle_x, middle_y), (thumb_x, thumb_y), (0, 0, 255), 2)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), 2)

            # Calculate finger lengths
            index_to_middle_length = math.hypot(middle_x - index_x, middle_y - index_y)
            middle_to_thumb_length = math.hypot(thumb_x - middle_x, thumb_y - middle_y)
            thumb_to_index_length = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Write the lengths on the lines
            cv2.putText(frame, f'{index_to_middle_length:.2f}', (int((index_x + middle_x) / 2), int((index_y + middle_y) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f'{middle_to_thumb_length:.2f}', (int((middle_x + thumb_x) / 2), int((middle_y + thumb_y) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f'{thumb_to_index_length:.2f}', (int((thumb_x + index_x) / 2), int((thumb_y + index_y) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Write the coordinates on the screen
            cv2.putText(frame, f'Thumb: ({thumb_x}, {thumb_y})', (thumb_x, thumb_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Index: ({index_x}, {index_y})', (index_x, index_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Middle: ({middle_x}, {middle_y})', (middle_x, middle_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add your conditions for PyAutoGUI actions
            if index_to_middle_length < 30:
                pyautogui.moveTo(((4/3) * index_x - (100/3)) * (screen_width / frame_width), ((4/3) * index_y - (100/3)) * (screen_height / frame_height))
            if thumb_to_index_length < 30:
                pyautogui.click()
                pyautogui.sleep(0.5)
            if middle_to_thumb_length < 30:
                pyautogui.mouseDown()
                pyautogui.moveTo(((4/3) * middle_x - (100/3)) * (screen_width / frame_width), ((4/3) * middle_y - (100/3)) * (screen_height / frame_height))
                # Check if the distance is greater than 30 to release the mouse button
                if middle_to_thumb_length > 30:
                    pyautogui.mouseUp()
                    pyautogui.sleep(0.5)

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