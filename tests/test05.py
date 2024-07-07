#Processing only the clicked frame

import cv2
import mediapipe as mp
from math import hypot

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load the hat image
hat_img = cv2.imread('acc_images/hat.png')
if hat_img is None:
    raise FileNotFoundError("Cannot load image at path: acc_images/hat.png")

# Initialize MediaPipe Face Mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

# Landmark indices for fitting the hat
hat_landmarks = [251, 334, 105, 21, 10]

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Display the current frame
    cv2.imshow("Press 'c' to capture", frame)

    # Wait for the 'c' key to be pressed to capture the frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Resize and convert the frame to RGB
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe Face Mesh
        results = faceMesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                leftx, lefty = 0, 0
                rightx, righty = 0, 0
                centerx, centery = 0, 0

                for lm_id, lm in enumerate(face_landmarks.landmark):
                    h, w, c = rgb.shape
                    x, y = int(lm.x * w), int(lm.y * h)

                    if lm_id == hat_landmarks[0]:
                        leftx, lefty = x, y
                    if lm_id == hat_landmarks[3]:
                        rightx, righty = x, y
                    if lm_id == hat_landmarks[4]:
                        centerx, centery = x, y

                hat_width = int(hypot(leftx - rightx, lefty - righty) * 1.8)
                hat_height = int(hat_width * 0.55)

                if hat_width > 0 and hat_height > 0:
                    hat_resized = cv2.resize(hat_img, (hat_width, hat_height))
                else:
                    print(f"Invalid hat dimensions: width={hat_width}, height={hat_height}")
                    continue

                top_left = (int(centerx - hat_width / 1.9), int(centery - hat_height / 1.35))
                bottom_right = (int(centerx + hat_width / 2), int(centery + hat_height / 4))

                if top_left[0] < 0 or top_left[1] < 0 or bottom_right[0] > w or bottom_right[1] > h:
                    print("Hat dimensions are out of frame bounds.")
                    continue

                hat_area = frame[top_left[1]: top_left[1] + hat_height, top_left[0]: top_left[0] + hat_width]
                if hat_area.size == 0:
                    print(f"Invalid hat area: top_left={top_left}, bottom_right={bottom_right}")
                    continue

                hat_gray = cv2.cvtColor(hat_resized, cv2.COLOR_BGR2GRAY)
                _, hat_mask = cv2.threshold(hat_gray, 25, 255, cv2.THRESH_BINARY_INV)

                hat_mask = cv2.resize(hat_mask, (hat_width, hat_height))
                hat_area = cv2.resize(hat_area, (hat_width, hat_height))

                if hat_area.shape[:2] != hat_mask.shape[:2]:
                    print("Mismatch in hat area and hat mask dimensions.")
                    continue

                if hat_area.dtype != hat_mask.dtype:
                    print("Mismatch in hat area and hat mask types.")
                    continue

                no_hat = cv2.bitwise_and(hat_area, hat_area, mask=hat_mask)
                final_hat = cv2.add(no_hat, hat_resized)

                frame[top_left[1]: top_left[1] + hat_height, top_left[0]: top_left[0] + hat_width] = final_hat

        # Display the output frame with the hat
        cv2.imshow("Output", frame)
        # Save the output frame to a file if needed
        cv2.imwrite("output_with_hat.png", frame)
    
    # Exit the loop if the 'q' key is pressed
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
