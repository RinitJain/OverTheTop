#Scalp Smart YOLO Model + Mediapipe


from ultralytics import YOLO
import cv2
import cvzone
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def detect_head(person_image_path):
    model = YOLO('model/version4_nanoyolo_20epoch_best.pt')
    results = model.predict(person_image_path)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            return [x1, y1, x2, y2]

def hat1(x1, y1, x2, y2):
    x_resize = int(1.3 * (x2 - x1))
    y_resize = int(1.3 * (y2 - y1))
    resize = [x_resize, y_resize]

    x_loc = int(x1 / 1.3 + (x2 - x1) / 11)
    y_loc = int(y1 / 1.3 - (y2 - y1) / 3)
    coordinates = [x_loc, y_loc]

    return [resize, coordinates]

def estimate_pose(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract important landmarks
            nose_tip = face_landmarks.landmark[1]  # Example: nose tip
            forehead = face_landmarks.landmark[10]  # Example: forehead
            # Convert normalized landmarks to pixel coordinates
            h, w, _ = image.shape
            nose_tip_coords = (int(nose_tip.x * w), int(nose_tip.y * h))
            forehead_coords = (int(forehead.x * w), int(forehead.y * h))
            return nose_tip_coords, forehead_coords
    return None, None

# Pass the path of person image here
person_image_loc = 'test_images/b3.jpg'
person_image = cv2.imread(person_image_loc)
# Pass the path of accessory image here
acc_image = cv2.imread('acc_images/hat1.png', cv2.IMREAD_UNCHANGED)

loc = detect_head(person_image_loc)
x1 = int(loc[0])
y1 = int(loc[1])
x2 = int(loc[2])
y2 = int(loc[3])

hat_type = int(input("1->Hat1: "))

if hat_type == 1:
    specs = hat1(x1, y1, x2, y2)

x_resize = specs[0][0]
y_resize = specs[0][1]
x_loc = specs[1][0]
y_loc = specs[1][1]

# Get pose estimation for better accuracy
nose_tip_coords, forehead_coords = estimate_pose(person_image_loc)
if nose_tip_coords and forehead_coords:
    x_loc = forehead_coords[0] - x_resize // 2
    y_loc = forehead_coords[1] - y_resize // 2

resized_acc_image = cv2.resize(acc_image, (x_resize, y_resize))

resultImg = cvzone.overlayPNG(person_image, resized_acc_image, [x_loc, y_loc])

# Display the result using matplotlib
plt.imshow(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
