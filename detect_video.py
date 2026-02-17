from ultralytics import YOLO
import cv2
import os


def detect_video(video_path, confidence=0.5, show_preview=False):

    model = YOLO('best.pt')
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    output_dir = 'output/video_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(video_path)}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    total_violations = 0
    total_safe = 0
    
    print("\nProcessing frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        results = model(frame, conf=confidence, verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = model.names[cls]
                
                if label == 'without_helmet':
                    total_violations += 1
                    color = (0, 0, 255)  
                    text = f'NO HELMET {conf_score:.2f}'
                else:
                    total_safe += 1
                    color = (0, 255, 0)  
                    text = f'with_helmet {conf_score:.2f}'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f'Frame: {frame_num}/{total_frames}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Violations: {total_violations}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'Safe: {total_safe}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        
        if show_preview:
            cv2.imshow('Processing...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress
        if frame_num % 100 == 0:
            print(f"  {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Video Processing Complete!")
    print(f"Output: {output_path}")
    print(f"Total Violations: {total_violations}")
    print(f"Safe Detections: {total_safe}")

if __name__ == "__main__":
    video_path = "F:/Universty/Bike Helmet Detection/bike_1.mp4"
    
    if os.path.exists(video_path):
        detect_video(video_path, confidence=0.5, show_preview=False)
    else:
        print(f"Error: Video not found at {video_path}")
        print("Please update the video_path variable")