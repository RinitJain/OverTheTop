#Purely MediaPipe


import cv2
import cvzone
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def estimate_pose(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract important landmarks
            forehead = face_landmarks.landmark[10]  # Example: forehead
            top_head = face_landmarks.landmark[9]   # Top of the head
            left_head = face_landmarks.landmark[234]  # Left side of the head
            right_head = face_landmarks.landmark[454]  # Right side of the head
            # Convert normalized landmarks to pixel coordinates
            
            h, w, _ = image.shape
            forehead_coords = (int(forehead.x * w), int(forehead.y * h))
            top_head_coords = (int(top_head.x * w), int(top_head.y * h))
            left_head_coords = (int(left_head.x * w), int(left_head.y * h) -10)
            right_head_coords = (int(right_head.x * w), int(right_head.y * h) +10)

            #print(forehead_coords, top_head_coords, left_head_coords, right_head_coords)

            return forehead_coords, top_head_coords, left_head_coords, right_head_coords
    return None, None, None, None

def place_hat(image, hat_image, forehead_coords, top_head_coords, left_head_coords, right_head_coords):
    # Calculate head width
    print("Right: ", right_head_coords)
    print("left: ", left_head_coords)

    head_width = right_head_coords[0] - left_head_coords[0]

    #2.875 is a hardcoded number
    #It is to accomodate the ring around the hat
    head_width = head_width + int(head_width/2.5)
    print("Head Width: ", head_width)
    
    # Resize the hat image to match the head width
    aspect_ratio = hat_image.shape[1] / hat_image.shape[0]
    print("Dimension: ", hat_image.shape)
    print("Aspect Ratio: ", aspect_ratio)

    hat_height = int((head_width / aspect_ratio))
    print("Hat Height: ", hat_height)

    resized_hat_image = cv2.resize(hat_image, (head_width, hat_height))
    
    # Calculate the position to place the hat
    x_offset = forehead_coords[0] - head_width // 2
    y_offset = top_head_coords[1] - hat_height

    return cvzone.overlayPNG(image, resized_hat_image, [x_offset, y_offset])

# Ensure the paths are correct
person_image_loc = 'test_images/b3.jpg'
person_image = cv2.imread(person_image_loc)
if person_image is None:
    raise FileNotFoundError(f"Person image not found at path: {person_image_loc}")

acc_image_loc = 'acc_images/hat1.png'
acc_image = cv2.imread(acc_image_loc, cv2.IMREAD_UNCHANGED)
if acc_image is None:
    raise FileNotFoundError(f"Accessory image not found at path: {acc_image_loc}")

# Get pose estimation for better accuracy
forehead_coords, top_head_coords, left_head_coords, right_head_coords = estimate_pose(person_image_loc)
if forehead_coords and top_head_coords and left_head_coords and right_head_coords:
    resultImg = place_hat(person_image, acc_image, forehead_coords, top_head_coords, left_head_coords, right_head_coords)

# Display the result using matplotlib
plt.imshow(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
