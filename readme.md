# Overall Steps to run:

## Step One:

```bash
git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2
```

## Step Two:

```bash
pip install -r requirement.txt
```

## Step Three:

```bash
bash main.sh "YOUR PDF DOCUMENT FILE PATH"
```

## Result

After running the script, the following directories will contain the generated output:

- **`figures/`**: Contains the figures extracted from the PDF.
- **`tables/`**: Contains the tables extracted from the PDF.
- **`output_images/`**: Stores the images of the original PDF pages.
- **`output_visualizations/`**: Contains the visualizations, such as segmentation results.

Check these folders for the converted images and visual outputs.

