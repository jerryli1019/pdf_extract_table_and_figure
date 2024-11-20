import argparse
import os
import glob
from ultralytics import YOLO

import cv2
from ditod import add_vit_config
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import shutil  # Import shutil to use rmtree

def check_and_create_folder(path):
    if os.path.exists(path):
        # Remove the entire directory tree
        shutil.rmtree(path)
    # Create the empty folder
    os.makedirs(path)
    
def main():
    parser = argparse.ArgumentParser(description="Detectron2 batch inference script")
    parser.add_argument(
        "--image_folder",
        help="Path to the folder containing input images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_folder",
        help="Root folder to store outputs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # Instantiate configuration
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Define model
    predictor = DefaultPredictor(cfg)
    
    # Prepare metadata
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0] == 'icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text", "title", "list", "table", "figure"])

    class_names = md.get("thing_classes", None)
    if class_names is None:
        print("No class names found in metadata.")
        return

    # Ensure the root output directory is empty
    check_and_create_folder(args.output_folder)

    # Initialize YOLO model
    model = YOLO('img_to_eqt/runs/detect/train9/weights/best.pt')
        
    # Process each image in the folder
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

        # Create a separate folder for each page under the root output folder
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        page_output_folder = os.path.join(args.output_folder, base_filename)
        check_and_create_folder(page_output_folder)  # Ensure the page folder is empty

        # YOLO Inference
        yolo_result = model(image_path)
        
        # Save YOLO detected images (equations) into the page's folder
        for idx, result in enumerate(yolo_result[0].boxes):
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cropped_image = img[y1:y2, x1:x2]
            cropped_image_path = os.path.join(page_output_folder, f'eqt_image_{idx}.png')
            cv2.imwrite(cropped_image_path, cropped_image)

        # Run Detectron2 inference
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")
        
        # Optional: Save overall visualization to the page's folder
        # v = Visualizer(img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        # result = v.draw_instance_predictions(predictions)
        # result_image = result.get_image()[:, :, ::-1]
        # output_visualization_path = os.path.join(page_output_folder, f"{base_filename}_vis.png")
        # cv2.imwrite(output_visualization_path, result_image)
        # print(f"Saved visualization to {output_visualization_path}")
        
        score = predictions.scores

        # Extract and save tables and figures into the page's folder
        for idx, (box, pred_class) in enumerate(zip(predictions.pred_boxes, predictions.pred_classes)):
            class_name = class_names[pred_class]
            # Only process tables and figures with a high confidence score
            if class_name in ["table", "figure"] and score[idx] > 0.90:
                bbox = box.numpy().astype(int)
                x0, y0, x1, y1 = bbox
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
                cropped_img = img[y0:y1, x0:x1]
                output_filename = os.path.join(page_output_folder, f"{class_name}_{idx}.png")
                cv2.imwrite(output_filename, cropped_img)
                print(f"Saved {class_name} to {output_filename}")

if __name__ == '__main__':
    main()
