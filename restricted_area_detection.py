#live count

import numpy as np
import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Define global variable for storing coordinates and the count of people in the restricted area
coordinates = []
people_count = 0

# Mouse callback function to collect coordinates
def draw(event, x, y, flags, params):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN and flags == 1:
        coordinates.append((x, y))
        if len(coordinates) == 4:
            print(coordinates)
            cv2.destroyAllWindows()

# Create window and load image for drawing coordinates
cv2.namedWindow(winname='res')
img = cv2.imread(r'C:\Users\ASUS\MACHINE LEARNING AND DATA SCIENCE\OPENCV\videos\frames\img0.jpg')
img = cv2.resize(img, (720, 480))

def callback_with_result(event, x, y, flags, params):
    draw(event, x, y, flags, params)
    return False

cv2.setMouseCallback("res", callback_with_result, None)
cv2.imshow("res", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)

# Open video for processing
video = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\vid.mp4")
restricted_area = coordinates
restricted_area_shapely = Polygon(restricted_area)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (720, 480))

    if not ret:
        break

    results = model(frame, classes=[0])
    boxes = results[0].boxes
    people_count = 0  # Reset count for each frame

    for box in boxes:
        class_id = int(box.numpy().cls[0])
        x1, y1, x2, y2 = box.numpy().xyxy[0]

        x3, y3 = x1 + abs(x2 - x1), y1
        x4, y4 = x1, y1 + abs(y1 - y2)

        restricted_area_np = np.array(restricted_area)
        cv2.polylines(frame, [restricted_area_np], True, (255, 0, 0), 4)

        person_polygon_shapely = Polygon([(x1, y1), (x4, y4), (x2, y2), (x3, y3)])
        intersection_area = restricted_area_shapely.intersection(person_polygon_shapely).area
        union_area = restricted_area_shapely.union(person_polygon_shapely).area
        iou = intersection_area / union_area if union_area > 0 else 0

        label1 = "Person"
        label2 = "Person_in_restricted_area"

        if model.names.get(class_id) == 'person':
            if iou > 0.01:
                people_count += 1  # Increment count if person is in restricted area
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label2, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label1, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the count of people in the restricted area
    cv2.putText(frame, f'People in restricted area: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Restricted area detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(people_count)
video.release()
cv2.destroyAllWindows()
