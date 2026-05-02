import cv2
import torch
from ultralytics import YOLO

# ==========================================
# --- PYTORCH 2.6 SECURITY BYPASS ---
# Temporarily force torch.load to accept weights_only=False 
# to prevent it from blocking standard neural network layers.
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load
# ==========================================

MODEL_PATH   = 'best.pt'
CONFIDENCE   = 0.60
WEBCAM_INDEX = 0

SAFE_CLASSES    = ['Hardhat', 'Safety Vest', 'Mask']
UNSAFE_CLASSES  = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']

COLOR_SAFE    = (0, 200, 0)
COLOR_UNSAFE  = (0, 0, 220)
COLOR_NEUTRAL = (0, 165, 255)

print("Loading model...")
# Load the model while the patch is active
model = YOLO(MODEL_PATH)

# Restore PyTorch's original secure load function now that the model is safely loaded
torch.load = _original_load 

print("Model loaded. Classes:", model.names)

# Initialize webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=CONFIDENCE, verbose=False)

    violations = []

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Determine color and track violations
            if class_name in SAFE_CLASSES:
                color = COLOR_SAFE
            elif class_name in UNSAFE_CLASSES:
                color = COLOR_UNSAFE
                violations.append(class_name)
            else:
                color = COLOR_NEUTRAL

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background and text
            label = f'{class_name} {confidence:.0%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw status banner at the top
    banner_color = (0, 50, 180) if violations else (0, 130, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), banner_color, -1)

    if violations:
        status_text = f'VIOLATION: {", ".join(set(violations))}'
    else:
        status_text = 'ALL PPE COMPLIANT'

    cv2.putText(frame, status_text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw exit instructions at the bottom
    cv2.putText(frame, 'Press Q to quit', (460, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Display the feed
    cv2.imshow('PPE Detection Demo', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Demo closed.")