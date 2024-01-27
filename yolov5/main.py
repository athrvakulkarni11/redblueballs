import cv2
import torch

# Load YOLOv5 model with force_reload=True
weights_path = "runs/train/exp2/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)


# Open the camera (change the argument to 0 if you have only one camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Display the results
    results.show()

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
