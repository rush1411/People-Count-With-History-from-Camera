import cv2
import numpy as np
import mysql.connector
import datetime
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO layer names
layer_names = net.getLayerNames()
i=[0]
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize camera
cap = cv2.VideoCapture(0)  

# Connect to MySQL database and create table
conn = mysql.connector.connect(
    host="localhost", user="root",
    password="", database="people_count")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS visits (
             id INT AUTO_INCREMENT PRIMARY KEY, 
             visit_time TIMESTAMP)''')
conn.commit()

def update_database():
    now = datetime.datetime.now()
    c.execute("INSERT INTO visits (visit_time) VALUES (%s)", (now,))
    conn.commit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Start time before processing
    start_time = time.time()  

    # Detect objects using YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Measure the time taken by forward pass
    forward_start = time.time()
    outs = net.forward(output_layers)
    forward_end = time.time()

    print(f"YOLO forward pass took {forward_end - forward_start:.2f} seconds")

    people_count = 0

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filter by 'person' class
                people_count += 1

                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw bounding box
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2),
                              (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)


    if people_count > 0:
        update_database()

    cv2.putText(frame, f'People Count: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end_time = time.time()  # End time after processing frame
    print(f"Frame processing took {end_time - start_time:.2f} seconds")


cap.release()
cv2.destroyAllWindows()
conn.close()
