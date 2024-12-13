# yoloV11_region_Adjust

Dynamic Region-Based Object Detection
This project focuses on dynamic, region-based object detection, allowing users to interactively adjust or define regions of interest (ROIs) using a mouse. This is particularly useful for tasks that require precision in selecting detection areas within a given frame or image.

Features
Interactive Region Adjustment: Modify existing regions of interest dynamically.
Custom Region Drawing: Define new regions of interest from scratch.
Built with Python and utilizes libraries such as Ultralytics for detection and OpenCV for graphical interaction.
Requirements
Ensure the following are installed before running the scripts:

Python 3.8 or newer
Libraries:
Ultralytics
OpenCV (cv2)
Install the required libraries using:

bash
Copy code
pip install ultralytics opencv-python  
Running the Project
Part 1: Adjust Regions Using Mouse
Modify existing regions of interest in an interactive session.

Command to Run:
bash
Copy code
python adjust_regions.py  
Description:
This script loads predefined regions and allows the user to adjust their boundaries using the mouse. It is ideal for refining ROIs in scenarios where precision is key.

Part 2: Draw Regions Using Mouse
Create custom regions of interest by drawing them interactively.

Command to Run:
bash
Copy code
python draw_regions.py  
Description:
This script enables users to draw new regions directly on the image or frame. These regions can then be used for targeted object detection tasks.

