from ultralytics import YOLO
import cv2

def detect_webcam(confidence=0.5, camera_index=0): 
    model = YOLO('best.pt')

    print("Starting Webcam Detection")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    frame_count = 0
    total_violations = 0
    total_safe = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = model(frame, conf=confidence, verbose=False)
        
        frame_violations = 0
        frame_safe = 0
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = model.names[cls]
                
                if label == 'without_helmet':
                    frame_violations += 1
                    total_violations += 1
                    color = (0, 0, 255)  
                    text = f'VIOLATION! {conf_score:.2f}'
                else:
                    frame_safe += 1
                    total_safe += 1
                    color = (0, 255, 0)  
                    text = f'SAFE {conf_score:.2f}'
            
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_count}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Violations: {total_violations}', (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f'Total Safe: {total_safe}', (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, 'Press Q to quit', (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Helmet Detection - Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("Webcam Detection Stopped")
    print(f"Total Frames: {frame_count}")
    print(f"Total Violations: {total_violations}")
    print(f"Total Safe: {total_safe}")

if __name__ == "__main__":
    detect_webcam(confidence=0.5, camera_index=0)