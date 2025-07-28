import os
import cv2
import tkinter as tk
from tkinter import messagebox
from src.inference import predict_frame

def main():
    cap = cv2.VideoCapture(0)
    root = tk.Tk()
    root.withdraw()  # hide TK window

    drowsy_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_frame(frame)
        text = f"{label} ({conf*100:.1f}%)"
        color = (0, 255, 0) if label=='Awake' else (0,0,255)
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if label=='Drowsy':
            drowsy_count += 1

        cv2.imshow('Drowsiness Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Pop-up summary
    sleepers = drowsy_count
    messagebox.showinfo("Session Summary",
        f"Drowsy detections: {sleepers}\n(Age estimation not implemented)")

if __name__ == "__main__":
    main()
