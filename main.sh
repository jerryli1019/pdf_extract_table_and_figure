#!/bin/bash

# Bash script to convert a PDF to images and run object detection inference

# Check if the PDF file path is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 path_to_pdf_file"
  exit 1
fi

PDF_PATH="$1"

# Check if the PDF file exists
if [ ! -f "$PDF_PATH" ]; then
  echo "Error: PDF file '$PDF_PATH' does not exist."
  exit 1
fi

# Optional: Activate your Python virtual environment
# Uncomment and modify the line below if you're using a virtual environment
# source /path/to/your/virtualenv/bin/activate

# Step 1: Convert PDF to images
echo "Converting PDF to images..."
python3 pdf2img.py --pdf_path "$PDF_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to convert PDF to images."
  exit 1
fi
echo "PDF conversion completed."

# Step 2: Run object detection inference
echo "Running object detection inference..."
python ./unilm/dit/object_detection/inference.py \
  --image_folder output_images \
  --output_folder output_visualizations \
  --config-file ./unilm/dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml \
  --opts MODEL.WEIGHTS publaynet_dit-b_mrcnn.pth
if [ $? -ne 0 ]; then
  echo "Error: Object detection inference failed."
  exit 1
fi
echo "Object detection inference completed."

echo "Processing completed successfully."
