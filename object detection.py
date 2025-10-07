from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt for better accuracy

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()