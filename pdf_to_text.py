from transformers import AutoModel, AutoTokenizer
import argparse
import os
import glob

def check_and_create_folder(path):
    if os.path.exists(path):
        # If folder exists, clear it by deleting its contents
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
    else:
        # If folder does not exist, create it
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Batch OCR extraction script")
    parser.add_argument(
        "--image_folder",
        help="Path to the folder containing input images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        help="Root folder to store output text files",
        type=str,
        required=True,
    )
    
    args = parser.parse_args()
    
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'ucaslcl/GOT-OCR2_0', 
        trust_remote_code=True, 
        low_cpu_mem_usage=True, 
        device_map='cuda', 
        use_safetensors=True, 
        pad_token_id=tokenizer.eos_token_id
    ).eval().cuda()
    
    # Ensure root output folder exists
    check_and_create_folder(args.output_folder)
    
    # Gather all image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))

    if not image_paths:
        print("No images found in the specified folder.")
        return

    # Process each image
    for image_path in image_paths:
        try:
            # Generate OCR result from model
            res = model.chat(tokenizer, image_path, ocr_type='format')
            
            # Create a unique folder for each page (image) under the root output folder
            page_name = os.path.splitext(os.path.basename(image_path))[0]
            page_output_folder = os.path.join(args.output_folder, page_name)
            os.makedirs(page_output_folder, exist_ok=True)
            
            # Define output text file path
            output_file_path = os.path.join(page_output_folder, f'{page_name}.txt')

            # Write text result to the file
            with open(output_file_path, 'w') as f:
                f.write(res)
                
            print(f"Text saved for {page_name} in {output_file_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    main()
