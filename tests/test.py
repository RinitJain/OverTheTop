#Using Scalp Smart YOLO model



from ultralytics import YOLO
import cv2
import cvzone
import matplotlib.pyplot as plt

#detects the head using the scalp smart model

def detect_head(person_image_path):

    model = YOLO('model/version4_nanoyolo_20epoch_best.pt')
    results = model.predict(person_image_path)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            return [x1,y1,x2,y2]
        
def hat1(x1,y1,x2,y2):
    #this will return an array which contains resize dimensions and position coordinates
    x_resize = int(1.3*(x2-x1))
    y_resize = int(1.3*(y2-y1))
    resize = [x_resize, y_resize]

    x_loc = int(x1/1.3 + (x2-x1)/11)
    y_loc = int(y1/1.3 - (y2-y1)/3)
    coordinates = [x_loc, y_loc]

    return [resize, coordinates]

        
        
#pass the path of person image here
person_image_loc = 'test_images/b3.jpg'
person_image = cv2.imread(person_image_loc)
#pass the path of accessory image here
acc_image = cv2.imread('acc_images/hat1.png', cv2.IMREAD_UNCHANGED)


loc = detect_head(person_image_loc)
x1 = int(loc[0])
y1 = int(loc[1])
x2 = int(loc[2])
y2 = int(loc[3])

hat_type = int(input("1->Hat1"))

if(hat_type==1):
    specs = hat1(x1,y1,x2,y2)

x_resize = specs[0][0]
y_resize = specs[0][1]
x_loc = specs[1][0]
y_loc = specs[1][1]

resized_acc_image = cv2.resize(acc_image, (x_resize,y_resize))

resultImg = cvzone.overlayPNG(person_image,resized_acc_image, [x_loc,y_loc])

#cv2.imshow('result',resultImg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()