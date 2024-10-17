import argparse
import os
import glob

import cv2
from ditod import add_vit_config
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

def main():
    parser = argparse.ArgumentParser(description="Detectron2 batch inference script")
    parser.add_argument(
        "--image_folder",
        help="Path to the folder containing input images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        help="Directory to save output visualizations (optional)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # Step 1: Instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: Add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: Define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: Prepare metadata
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0] == 'icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text", "title", "list", "table", "figure"])

    class_names = md.get("thing_classes", None)
    if class_names is None:
        print("No class names found in metadata.")
        return

    # Step 6: Create output directories if they don't exist
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    # Step 7: Process each image in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))

    if not image_paths:
        print("No images found in the specified folder.")
        return

    for image_path in image_paths:
        print(f"Processing {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image {image_path}")
            continue

        # Run inference
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")

        # Optional: Save overall visualization
        if args.output_folder:
            v = Visualizer(img[:, :, ::-1],
                           md,
                           scale=1.0,
                           instance_mode=ColorMode.SEGMENTATION)
            result = v.draw_instance_predictions(predictions)
            result_image = result.get_image()[:, :, ::-1]
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_visualization_path = os.path.join(args.output_folder, f"{base_filename}_vis.png")
            cv2.imwrite(output_visualization_path, result_image)
            print(f"Saved visualization to {output_visualization_path}")

        # Extract and save tables and figures
        for idx, (box, pred_class) in enumerate(zip(predictions.pred_boxes, predictions.pred_classes)):
            class_name = class_names[pred_class]
            # Only process tables and figures
            if class_name in ["table", "figure"]:
                # Convert tensor to numpy array and get coordinates
                bbox = box.numpy().astype(int)
                x0, y0, x1, y1 = bbox
                # Ensure coordinates are within image boundaries
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(img.shape[1], x1)
                y1 = min(img.shape[0], y1)
                # Crop the image
                cropped_img = img[y0:y1, x0:x1]
                # Determine the output directory
                output_dir = 'tables' if class_name == 'table' else 'figures'
                # Create a unique filename
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = os.path.join(output_dir, f"{base_filename}_{class_name}_{idx}.png")
                # Save the cropped image
                cv2.imwrite(output_filename, cropped_img)
                print(f"Saved {class_name} to {output_filename}")

if __name__ == '__main__':
    main()


