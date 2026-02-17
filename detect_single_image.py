from ultralytics import YOLO
import cv2
import os

def detect_image(image_path, confidence=0.5):
    
    model = YOLO('best.pt')
    image_path = os.path.abspath("F:/Universty/Bike Helmet Detection/image10342.jpg")
    print(f"Processing: {image_path}")
    
    results = model.predict(
        source=image_path,
        conf=confidence,
        save=True,
        project='output',
        name='image_results',
        exist_ok=True
    )
    
    print("\nDetections:")
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"  {label}: {conf:.2f}")
    
    print("\n Results saved in: Single_Image_output/image_results/")
    return results

if __name__ == "__main__":
    # Test on an image
    image_path = "F:/Universty/Bike Helmet Detection/image10342.jpg"
    
    if os.path.exists(image_path):
        detect_image(image_path, confidence=0.5)
    else:
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path variable")