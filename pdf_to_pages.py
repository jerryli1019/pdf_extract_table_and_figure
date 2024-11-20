import fitz
import os
import argparse
import cv2
import layoutparser as lp
import shutil  # Import shutil to use rmtree

def check_and_create_folder(path):
    if os.path.exists(path):
        # If folder exists, remove it entirely
        shutil.rmtree(path)
    # Create the empty folder
    os.makedirs(path)
            
def process_pdf(pdf_path, output_folder):
    check_and_create_folder(output_folder)
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f'page_{page_number + 1}.png')
        pix.save(image_path)
        image_paths.append(image_path)
        print(f'Saved {image_path}')
    
def main():
    parser = argparse.ArgumentParser(description='Process a PDF file to extract tables, figures, and equations.')
    parser.add_argument('-p', '--pdf_path', default='transformer.pdf', help='Path to the PDF file')
    parser.add_argument('-o', '--output', default='output_pages', help='Output directory for the images')
    args = parser.parse_args()
    process_pdf(args.pdf_path, args.output)
    
if __name__ == "__main__":
    main()
