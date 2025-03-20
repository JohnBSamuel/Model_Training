import cv2
import os

# Paths
dataset_path = "/Users/johnsamuel/Desktop/Civilens_Dataset"
image_folder = os.path.join(dataset_path)  # No 'images/' subfolder
label_folder = os.path.join(dataset_path, "labels")

# Ensure labels directory exists
os.makedirs(label_folder, exist_ok=True)

# Get list of images
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]
if not image_files:
    raise FileNotFoundError(f"❌ No images found in {image_folder}")

# Start with the first image
current_index = 0
image_path = os.path.join(image_folder, image_files[current_index])

# Load Image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")

height, width, _ = image.shape
bboxes = []

# Mouse Callback
drawing = False
x_start, y_start = -1, -1

def draw_bbox(event, x, y, flags, param):
    global x_start, y_start, drawing, bboxes, image

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # Convert to YOLO format
        x_center = (x_start + x_end) / (2 * width)
        y_center = (y_start + y_end) / (2 * height)
        bbox_width = abs(x_end - x_start) / width
        bbox_height = abs(y_end - y_start) / height

        # Save bbox (Class ID = 0 for No Parking sign)
        bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

# Initialize OpenCV Window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_bbox)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1)

    if key == ord("s"):  # Save annotations
        label_path = os.path.join(label_folder, f"{os.path.splitext(image_files[current_index])[0]}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(bboxes))
        print(f"✅ Annotations saved: {label_path}")

        # Move to next image
        current_index += 1
        if current_index >= len(image_files):
            print("🎉 All images annotated!")
            break
        
        # Load next image
        image_path = os.path.join(image_folder, image_files[current_index])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")
        height, width, _ = image.shape
        bboxes = []  # Reset bounding boxes

    elif key == ord("q"):  # Quit
        print("🚪 Exiting without saving.")
        break

cv2.destroyAllWindows()
