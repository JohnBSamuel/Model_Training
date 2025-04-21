import os
import cv2

# Update paths to your augmented images and labels
aug_img_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images_aug2"
aug_lbl_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels_aug2"
output_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/verify_bboxes"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(aug_img_dir):
    if not fname.endswith(('.jpg', '.jpeg', '.png')):
        continue

    name_wo_ext = os.path.splitext(fname)[0]
    img_path = os.path.join(aug_img_dir, fname)
    label_path = os.path.join(aug_lbl_dir, f"{name_wo_ext}.txt")

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        print(f"⚠️ No label file for {fname}")
        continue

    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{int(cls)}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Save the image with bounding boxes drawn for verification
    cv2.imwrite(os.path.join(output_dir, fname), image)

print("✅ Verification images with bounding boxes saved.")
