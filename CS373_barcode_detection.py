# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    def __iter__(self):
        return iter(self.items)

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def computeRGBToGreyscale(image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            r = pixel_array_r[y][x]
            g = pixel_array_g[y][x]
            b = pixel_array_b[y][x]
            g = round(0.299 * r + 0.587 * g + 0.114 * b)
            greyscale_pixel_array[y][x] = int(g)
    return greyscale_pixel_array

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    output_image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)
    for y in range(2, image_height - 2):
        for x in range(2, image_width - 2):
            kernel = []
            for j in range(-2, 3):
                for i in range(-2, 3):
                    p = pixel_array[y + j][x + i]
                    kernel.append(p)
            mean = sum(kernel) / 25.0
            v = sum((pixel - mean) ** 2 for pixel in kernel) / 25.0
            output_image[y][x] = math.sqrt(v)
    return output_image


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height, recursion_count=0):
    output_image = createInitializedGreyscalePixelArray(image_width, image_height)
    gaussian_kernel = [[1, 2, 1],[2, 4, 2],[1, 2, 1]]
    
    if recursion_count == 9:
        return pixel_array
    
    for y in range(image_height):
        for x in range(image_width):
            sum1 = 0
            for j in range(-1, 2):
                for i in range(-1, 2):
                    y1 = min(max(y + j, 0), image_height - 1)
                    x1 = min(max(x + i, 0), image_width - 1)
                    pix = pixel_array[y1][x1]
                    kernel = gaussian_kernel[j + 1][i + 1]
                    sum1 += pix * kernel
            final_x = sum1 / 16.0
            output_image[y][x] = final_x
    return computeGaussianAveraging3x3RepeatBorder(output_image, image_width, image_height, recursion_count + 1)

def computeThresholdGE(pixel_array, image_width, image_height, threshold_value=18):
    binary_image = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] < threshold_value:
                binary_image[y][x] = 0
            else:
                binary_image[y][x] = 255
    return binary_image

def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height, recursion_count=0):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    if recursion_count == 5:
        return pixel_array
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] > 0:
                forecheck = True
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        x1 = x + j
                        y1 = y + i
                        if 0 <= x1 < image_width and 0 <= y1 < image_height:
                            if pixel_array[y1][x1] == 0:
                                forecheck = False
                                break
                        else:
                            forecheck = False
                            break
                    if not forecheck:
                        break
                if forecheck:
                    output[y][x] = 1
    return computeErosion8Nbh5x5FlatSE(output, image_width, image_height, recursion_count + 1)

def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height, recursion_count=0):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    if recursion_count == 3:
        return pixel_array
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] > 0:
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        x1 = x + j
                        y1 = y + i
                        if 0 <= x1 < image_width and 0 <= y1 < image_height:
                            output[y1][x1] = 1
    return computeDilation8Nbh5x5FlatSE(output, image_width, image_height, recursion_count+1)

def computeConnectedComponentLargestXY(pixel_array, image_width, image_height):
    c = 0
    s = {}
    largest_size = 0
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    queue = Queue()
    fin_min_x = image_width
    fin_max_x = 0
    fin_min_y = image_height
    fin_max_y = 0
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] > 0 and output[y][x] == 0:
                c += 1
                queue.enqueue((x, y))
                current_size = 0
                min_x = image_width
                max_x = 0
                min_y = image_height
                max_y = 0
                while not queue.isEmpty():
                    x1, y1 = queue.dequeue()
                    if 0 <= x1 < image_width and 0 <= y1 < image_height and pixel_array[y1][x1] > 0:
                        if output[y1][x1] == 0:
                            output[y1][x1] = c
                            s[c] = s.get(c, 0) + 1
                            current_size += 1
                            min_x = min(min_x, x1)
                            max_x = max(max_x, x1)
                            min_y = min(min_y, y1)
                            max_y = max(max_y, y1)
                            queue.enqueue((x1 - 1, y1))
                            queue.enqueue((x1 + 1, y1))
                            queue.enqueue((x1, y1 - 1))
                            queue.enqueue((x1, y1 + 1))
                if current_size > largest_size and (max_x - min_x) / (max_y - min_y) <= 1.8:
                    largest_size = current_size
                    fin_min_x = min_x
                    fin_max_x = max_x
                    fin_min_y = min_y
                    fin_max_y = max_y
    return fin_min_x, fin_max_x, fin_min_y, fin_max_y

def separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]
    for y in range(image_height):
             for x in range(image_width):
                new_array[y][x][0] = px_array_r[y][x]
                new_array[y][x][1] = px_array_g[y][x]
                new_array[y][x][2] = px_array_b[y][x]
    return new_array





# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():


    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode5"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(1, 1)
    
    # STUDENT IMPLEMENTATION here
    greyscale = computeRGBToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    stdev = computeStandardDeviationImage5x5(greyscale, image_width, image_height)
    gaussianimage = computeGaussianAveraging3x3RepeatBorder(stdev, image_width, image_height)
    binaryimage = computeThresholdGE(gaussianimage, image_width, image_height)
    erodedimage = computeErosion8Nbh5x5FlatSE(binaryimage, image_width, image_height)
    dilatedimage = computeDilation8Nbh5x5FlatSE(erodedimage, image_width, image_height)
    tuple_output = computeConnectedComponentLargestXY(dilatedimage, image_width, image_height)

    px_array = separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    bbox_min_x = tuple_output[0]
    bbox_max_x = tuple_output[1] 
    bbox_min_y = tuple_output[2] 
    bbox_max_y = tuple_output[3] 
    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1.set_title('Final image of detection')
    axs1.imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1.add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()