from PIL import Image
from pix2tex.cli import LatexOCR


def main():
    img = Image.open('output_visualizations/page_3/eqt_image_1.png')
    model = LatexOCR()
    print(model(img))


if __name__ == '__main__':
    main()