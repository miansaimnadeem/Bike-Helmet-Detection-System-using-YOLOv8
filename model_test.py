from ultralytics import YOLO

model = YOLO('best.pt')
print("Model loaded successfully!")

print(f"\nModel: {model.model_name}")
print(f"Classes: {model.names}")
print(f"Number of classes: {len(model.names)}")

print("\nModel is ready to use!")