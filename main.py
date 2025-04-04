import cv2  # OpenCV for image processing...
import os  # OS module for file operations..
from ultralytics import YOLO  # YOLO model import.


def detect_helmet(image_path, model):  # Detect helmets in image..
    image = cv2.imread(image_path)  # Read image...
    if image is None:  # Check if image is loaded..
        print(f"Error: Could not read image {image_path}")  # Print error..
        return  # Exit function..

    results = model(image)  # Run YOLO model...

    for result in results:  # Iterate detection results..
        for box in result.boxes:  # Iterate detected objects..
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates...
            conf = float(box.conf[0])  # Get confidence score..
            label = result.names[int(box.cls[0])]  # Get label...

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box..
            cv2.putText(image, f'{label} ({conf:.2f})', (x1, y1 - 10),  # Add label..
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Set font...

    cv2.imshow("Helmet Detection", image)  # Show image..
    cv2.waitKey(0)  # Wait for key press...
    cv2.destroyAllWindows()  # Close windows..


def main():  # Main function..
    model_path = "models/yolo11_helmetdetection.pt"  # Model path...
    image_folder = "data"  # Image folder..

    if not os.path.exists(model_path):  # Check model file..
        print("Error: Model file not found.")  # Print error...
        return  # Exit..

    model = YOLO(model_path)  # Load YOLO model...

    if not os.path.exists(image_folder):  # Check image folder..
        print("Error: Image folder not found.")  # Print error..
        return  # Exit...

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Get images..
    if not image_files:  # Check if images exist...
        print("Error: No images found in the folder.")  # Print error..
        return  # Exit...

    for image_file in image_files:  # Loop images..
        detect_helmet(os.path.join(image_folder, image_file), model)  # Process image...


if __name__ == "__main__":  # Entry point..
    main()  # Run main function...
