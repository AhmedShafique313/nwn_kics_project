import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_and_annotate():
    # Open the HD USB external camera 
    cap = cv2.VideoCapture(1)  # Change the index (e.g., 0, 1, 2, etc.) based on your system configuration

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Draw lines on the frame
        cv2.line(frame, pt1=(220, 25), pt2=(220, 400), color=(0, 0, 255), thickness=2)
        cv2.line(frame, pt1=(430, 25), pt2=(430, 400), color=(0, 0, 255), thickness=2)

        # Add text annotations
        cv2.putText(frame, 'Person_1', (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Person_2', (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Person_3', (440, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the annotated frame
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC key to exit
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE key to capture and annotate
            print("Space hit, capturing and annotating...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    capture_and_annotate()