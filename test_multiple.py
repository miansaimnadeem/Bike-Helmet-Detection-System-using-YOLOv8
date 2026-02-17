import torch
from ultralytics import YOLO
import os

model = YOLO("best.pt")  

# Source folder 
source_path = r"F:\Universty\Bike Helmet Detection\Bike_Helmet_Detection_System\test\images"

# Output directory
output_path = r"F:\Universty\Bike Helmet Detection\Bike_Helmet_Detection_System\test_results"

results = model.predict(
    source=source_path,
    conf=0.5,
    save=True,
    project=output_path,
    name="predict"  
)

print("Detection completed successfully!")
print(f"Results saved in: {os.path.join(output_path, 'predict')}")