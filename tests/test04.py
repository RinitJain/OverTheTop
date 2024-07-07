import cv2
import mediapipe as mp
from math import hypot

cap = cv2.VideoCapture(0)#'test_images/1.mp4'

hat_img = cv2.imread('acc_images/hat.png')  # Load the hat image

# Check if the hat image was loaded correctly
if hat_img is None:
    raise FileNotFoundError("Cannot load image at path: acc_images/hat1.jpg")

prevTime = 0

# Landmark indices for fitting the hat: 10 = center forehead
# Line A (i.e. Above eyebrow): 251, 334, 105, 21
# Line B (i.e. Above Line A ): 284, 299, 69, 54
hat_landmarks = [251, 334, 105, 21, 10]

# Total 468 landmarks
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)  # max_num_faces=3

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # Resize the frame (640, 480), (1280, 720), (1920, 1080)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB

    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            leftx, lefty = 0, 0
            rightx, righty = 0, 0
            centerx, centery = 0, 0
            
            for lm_id, lm in enumerate(face_landmarks.landmark):
                h, w, c = rgb.shape
                x, y = int(lm.x * w), int(lm.y * h)

                # Face/Foreheads Landmarks considered
                # if lm_id in hat_landmarks:
                #     cv2.putText(
                #         frame, 
                #         str(lm_id),
                #         (x,y),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.4,
                #         (0,0,255),
                #         1
                #     )
                
                if lm_id == hat_landmarks[0]:
                    leftx, lefty = x, y
                if lm_id == hat_landmarks[3]:
                    rightx, righty = x, y
                if lm_id == hat_landmarks[4]:
                    centerx, centery = x, y
                
            hat_width = int(hypot(leftx - rightx, lefty - righty) * 1.8)  # Adjust factor as needed
            hat_height = int(hat_width * 0.55)  # Adjust this ratio based on the hat image aspect ratio

            if hat_width > 0 and hat_height > 0:
                hat_resized = cv2.resize(hat_img, (hat_width, hat_height))
            else:
                print(f"Invalid hat dimensions: width={hat_width}, height={hat_height}")
                continue
            
            top_left = (int(centerx - hat_width / 1.9), int(centery - hat_height/ 1.35))
            bottom_right = (int(centerx + hat_width / 2), int(centery + hat_height / 4))

            # The Rectangle where the hat will be placed
            # cv2.rectangle(
            #     frame, 
            #     top_left, 
            #     bottom_right, 
            #     (0, 255, 0), 
            #     1
            # )

            # When Hat Area goes out of bounds
            if top_left[0] < 0 or top_left[1] < 0 or bottom_right[0] > w or bottom_right[1] > h:
                print("Hat dimensions are out of frame bounds.")
                continue
            
            #Creating the hat area based on the rectangle
            hat_area = frame[
                top_left[1]: top_left[1] + hat_height, 
                top_left[0]: top_left[0] + hat_width
            ]

            if hat_area.size == 0:
                print(f"Invalid hat area: top_left={top_left}, bottom_right={bottom_right}")
                continue

            #Convert Image to Grayscale
            hat_gray = cv2.cvtColor(hat_resized, cv2.COLOR_BGR2GRAY)

            #Create a Mask of that Hat 
            _, hat_mask = cv2.threshold(hat_gray, 25, 255, cv2.THRESH_BINARY_INV)

            #Resize both the hat_mask and hat_area so that there is no Mismatch while fitting
            hat_mask = cv2.resize(hat_mask, (hat_width, hat_height))
            hat_area = cv2.resize(hat_area, (hat_width, hat_height))

            if hat_area.shape[:2] != hat_mask.shape[:2]:
                print("Mismatch in hat area and hat mask dimensions.")
                continue

            if hat_area.dtype != hat_mask.dtype:
                print("Mismatch in hat area and hat mask types.")
                continue

            #Remove the part of the head where hat is to be placed
            no_hat = cv2.bitwise_and(hat_area, hat_area, mask=hat_mask)

            #Add the final Hat
            final_hat = cv2.add(no_hat, hat_resized)

            #Creating final frame
            frame[
                top_left[1]: top_left[1] + hat_height, 
                top_left[0]: top_left[0] + hat_width
            ] = final_hat


    cv2.imshow("output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()