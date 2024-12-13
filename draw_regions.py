import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8l.pt")  # Replace with your YOLO model file

# Initialize variables for regions and drawing
regions = []
current_region = []
drawing = False

# Mouse callback function for drawing regions
def draw_region(event, x, y, flags, param):
    global drawing, current_region, regions
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_region = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_region.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_region.append((x, y))
        regions.append(np.array(current_region, dtype=np.int32))
        current_region = []

# Function to check if a point is inside any region
def is_point_in_region(point, regions):
    for region in regions:
        if cv2.pointPolygonTest(region, point, False) >= 0:
            return True
    return False

# Function to draw regions on the frame
def draw_regions(frame, regions):
    for region in regions:
        cv2.polylines(frame, [region], isClosed=True, color=(0, 255, 0), thickness=2)

# Open video capture
cap = cv2.VideoCapture("/home/sct/Downloads/vechicle_dataset1/Test Video.mp4")  # Replace with your video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Final2.mp4", fourcc, 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

cv2.namedWindow("Region-Based Detection")
cv2.setMouseCallback("Region-Based Detection", draw_region)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)  # YOLO inference

    # Filter detections based on regions
    filtered_boxes = []
    for result in results:
        for box in result.boxes:  # Iterate over detected objects
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Compute the center of the bounding box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Check if the center is in any region
            if is_point_in_region(center, regions):
                filtered_boxes.append((x1, y1, x2, y2, conf, cls))

    # Draw regions and current region
    draw_regions(frame, regions)
    if current_region:
        cv2.polylines(frame, [np.array(current_region, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

    # Annotate frame with detections
    for box in filtered_boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # Add title "Manual Adjust Area" to the top-left corner
    cv2.putText(frame, "Manual Draw Area", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Region-Based Detection", frame)

    # Write frame to output video
    out.write(frame)

    # Press 'q' to quit, 'c' to clear regions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        regions = []  # Clear all regions

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
