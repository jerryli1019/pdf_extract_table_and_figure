import fitz
import os
import argparse
import cv2
import layoutparser as lp

def process_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
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
    parser.add_argument('-p', '--pdf_path', default='transformer.pdf', help='Directory for the PDF')
    parser.add_argument('-o', '--output', default='output_images', help='Output directory for the images')
    args = parser.parse_args()
    process_pdf(args.pdf_path, args.output)

if __name__ == "__main__":
    main()
