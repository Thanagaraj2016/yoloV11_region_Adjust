import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8l.pt")  # Replace with your YOLO model file

# Function to initialize the region at the center with 20% of image size
def initialize_region(frame):
    h, w = frame.shape[:2]
    region_width = int(w * 0.2)  # 20% of the width
    region_height = int(h * 0.2)  # 20% of the height

    # Calculate the center of the frame
    center_x, center_y = w // 2, h // 2

    # Define the 4 vertices of the polygon (rectangle initially)
    top_left = (center_x - region_width // 2, center_y - region_height // 2)
    top_right = (center_x + region_width // 2, center_y - region_height // 2)
    bottom_right = (center_x + region_width // 2, center_y + region_height // 2)
    bottom_left = (center_x - region_width // 2, center_y + region_height // 2)

    # Return the region as a polygon
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

# Initialize region and mouse control variables
selected_point = None  # No point selected initially
dragging = False  # Track if a point is being dragged
radius = 10  # Radius for selecting a point

# Mouse callback function to adjust region (resize or reshape polygon)
def mouse_callback(event, x, y, flags, param):
    global selected_point, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the user clicked near any vertex (within the radius)
        for i, (px, py) in enumerate(region):
            if np.sqrt((x - px)**2 + (y - py)**2) < radius:
                selected_point = i  # Store the index of the selected point
                dragging = True
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_point is not None:
            # Update the selected point's position
            region[selected_point] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_point = None

# Function to draw the region (polygon)
def draw_region(frame, region):
    cv2.polylines(frame, [region], isClosed=True, color=(0, 255, 0), thickness=2)
    # Draw points (vertices) to indicate where the user can click
    for (x, y) in region:
        cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)

# Function to check if a point is inside the region
def is_point_in_region(point, region):
    return cv2.pointPolygonTest(region, point, False) >= 0

# Open video capture
cap = cv2.VideoCapture("/home/sct/Downloads/vechicle_dataset1/Test Video.mp4")  # Replace with your video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Final1.mp4", fourcc, 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

cv2.namedWindow("Region-Based Detection")
cv2.setMouseCallback("Region-Based Detection", mouse_callback)

# Initialize the region
ret, frame = cap.read()
if ret:
    region = initialize_region(frame)  # Set the initial region

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Filter detections based on the region
    filtered_boxes = []
    for result in results:
        for box in result.boxes:  # Iterate over detections
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Compute the center of the bounding box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Check if the center is inside the region
            if is_point_in_region(center, region):
                filtered_boxes.append((x1, y1, x2, y2, conf, cls))

    # Draw the region on the frame
    draw_region(frame, region)

    # Annotate frame with detections
    for box in filtered_boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # Add title "Manual Adjust Area" to the top-left corner
    cv2.putText(frame, "Manual Adjust Area", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Region-Based Detection", frame)

    # Write the frame to output video
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
