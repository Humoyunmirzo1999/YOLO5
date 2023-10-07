import torch
import cv2
import numpy as np

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Video
video_path = "video.mp4"  # Replace with your video file path
output_path = "output_video.mp4"  # Output video path

# Open the video file for reading
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Get the width and height of the video frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # Read a frame from the video

    if not ret:
        break  # Break the loop if we have reached the end of the video

    # Inference
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    output_frame = results.render()[0]

    # Convert the frame from BGR to RGB (OpenCV uses BGR by default)
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    # Write the frame to the output video
    out.write(output_frame)

    # Display the frame (optional, you can comment this out if not needed)
    cv2.imshow('YOLOv5 Object Detection', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
