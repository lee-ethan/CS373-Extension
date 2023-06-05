from flask import Flask, render_template, request
import math
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import imageIO.png
from CS373_barcode_detection import *

# Initialize Flask application
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for barcode detection
@app.route('/detect', methods=['POST'])
def detect_barcode():
    # Get uploaded image file
    image_file = request.files['image']
    image_filename = image_file.filename
    image_path = f"uploads/{image_filename}"
    image_file.save(image_path)

    # Call barcode detection function
    bbox = detect_barcode_in_image(image_path)

    # Generate output image with barcode outline
    output_path = f"static/output/{image_filename}"
    generate_output_image(image_path, bbox, output_path)

    # Render template with the output image
    return render_template('result.html', image_filename=image_filename, output_path=output_path)

# Function to detect the barcode in the image
def detect_barcode_in_image(image_path):
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(image_path)
    greyscale = computeRGBToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    stdev = computeStandardDeviationImage5x5(greyscale, image_width, image_height)
    gaussianimage = computeGaussianAveraging3x3RepeatBorder(stdev, image_width, image_height)
    binaryimage = computeThresholdGE(gaussianimage, image_width, image_height)
    erodedimage = computeErosion8Nbh5x5FlatSE(binaryimage, image_width, image_height)
    dilatedimage = computeDilation8Nbh5x5FlatSE(erodedimage, image_width, image_height)
    b_boxes = computeConnectedComponentLargestXY(dilatedimage, image_width, image_height)
    return b_boxes
# Function to generate an output image with the barcode outline
def generate_output_image(input_image_path, bbox, output_path):
    # Read the input image
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_image_path)
    px_array = separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)


    # Create a figure for plotting
    fig, ax = pyplot.subplots()

    # Display the input image
    ax.imshow(px_array, cmap='gray')

    # Draw the barcode bounding box
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = bbox
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # Save the output image
    pyplot.savefig(output_path)

# Remaining code...

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
