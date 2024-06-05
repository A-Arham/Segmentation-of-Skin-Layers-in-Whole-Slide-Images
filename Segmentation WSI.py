import numpy as np
import cv2


def removebg(image_matrix):

    gray_image = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    rows, cols, _ = image_matrix.shape


    removed_bg_image = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Iterate through each pixel in the binary mask
    for i in range(rows):
        for j in range(cols):
            # If the pixel in the mask is white (255), set the corresponding pixel in the new image
            if binary_mask[i][j] == 255:
                removed_bg_image[i][j] = image_matrix[i][j]

    return removed_bg_image

def colorize_binary_image(binary_image, color_code):

    # Ensure the input image is binary

    rows,cols=np.shape(binary_image)

    # Create an empty color image with the same dimensions
    colorized_image = np.zeros((rows,cols,3), dtype=np.uint8)

    # Assign the specified color to white pixels
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 255:
                colorized_image[i, j] = color_code

    return colorized_image

def find_pixels_by_color(image, target_color_bgr, tolerance=20):
    rows, cols, _ = image.shape
    target_color_bgr = np.array(target_color_bgr, dtype=np.uint8)

    matching_pixels = []

    for i in range(rows):
        for j in range(cols):
            pixel_color = image[i, j, :]
            color_difference = np.linalg.norm(pixel_color - target_color_bgr)

            if color_difference <= tolerance:
                matching_pixels.append((i, j))

    return matching_pixels

def merge_images(images):

    # Ensure all images have the same dimensions
    height, width, _ = images[0].shape

    # Create a blank canvas for the merged image with an alpha channel
    merged_image = np.zeros((height, width, 4), dtype=np.uint8)

    for i, image in enumerate(images):
        # Ensure the image has the same dimensions
        image = cv2.resize(image, (width, height))

        # Create an alpha channel (transparency) based on the image intensity
        alpha_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha_channel = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]

        # Set the RGB channels of the merged image using the current image and alpha channel
        merged_image[:, :, :3] = merged_image[:, :, :3] * (1 - alpha_channel // 255)[:, :, np.newaxis] + image[:, :, :3] * (alpha_channel // 255)[:, :, np.newaxis]

    return merged_image

def get_spectrum_of_most_repeated_bgr(image, coordinates,):

    bgr_values = [tuple(image[y, x].tolist()) for x, y in coordinates]

    unique_bgr_values, counts = np.unique(bgr_values, return_counts=True, axis=0)

    # Calculate the mean for each color channel separately
    average_bgr = tuple(np.mean(unique_bgr_values, axis=0).astype(int))


    return average_bgr

def mask_image_on_bgr(bgr_image, target_bgr, tolerance=25):

    lower_bound = np.array([max(0, val - tolerance) for val in target_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, val + tolerance) for val in target_bgr], dtype=np.uint8)

    # Create a binary mask within the specified tolerance range
    mask = cv2.inRange(bgr_image, lower_bound, upper_bound)

    return mask

def remove_non_listed_pixels(image, listed_coordinates):

    rows, cols, _ = image.shape

    # Create a mask with all pixels initially set to 0
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # Set pixels in the mask for listed coordinates to 255
    for x, y in listed_coordinates:
        mask[x, y] = 255

    # Set color values to [0, 0, 0] for pixels not in the list
    image[mask == 0] = [0, 0, 0]

    return image

def replace_black_spots(image, replacement_color, threshold=5):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask of black spots
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw filled contours with the replacement color
    result = image.copy()
    for contour in contours:
        cv2.drawContours(result, [contour], -1, replacement_color, thickness=cv2.FILLED)

    return result

def dice_cof(im1, im2):
    rows, cols, _ = im1.shape  # Assuming images are color (BGR)

    counter = 0
    for i in range(rows):
        for j in range(cols):
            # Check if shapes are compatible for element-wise comparison
            if ((im1[i, j,1] == im2[i, j,1])):
                counter += 1

    dim = rows * cols

    coefficient = counter / dim
    return coefficient

imgg=cv2.imread('tissue (1).jpg') # tissue img in bgr
imgg=cv2.blur(imgg,(3,3))

img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY) # tissue img in grey

mask = cv2.imread('mask (1).png')

rem=removebg(imgg)




pink=255,172,255
green=0,255,190
purple=160,48,112
grey=224,224,224


pink_list=find_pixels_by_color(mask,pink)
green_list=find_pixels_by_color(mask,green)
purple_list=find_pixels_by_color(mask,purple)
grey_list=find_pixels_by_color(mask,grey)

d=get_spectrum_of_most_repeated_bgr(imgg,green_list)
print("avg dermis:")
print(d)

e=get_spectrum_of_most_repeated_bgr(imgg,purple_list)
print("avg epidermis:")
print(e)

j=get_spectrum_of_most_repeated_bgr(imgg,pink_list)
print("avg epidermis:")
print(j)


k=get_spectrum_of_most_repeated_bgr(imgg,grey_list)
print("avg epidermis:")
print(k)

dermis_bin=mask_image_on_bgr(imgg,d)


epidermis_binary=mask_image_on_bgr(imgg,e)

jun_bin=mask_image_on_bgr(imgg,j)


keratin_binary=mask_image_on_bgr(imgg,k)

jun_bin_color=colorize_binary_image(jun_bin,pink)
keratin_binary_color=colorize_binary_image(keratin_binary,grey)
dermis_bin_color=colorize_binary_image(dermis_bin,green)
epidermis_binary_color=colorize_binary_image(epidermis_binary,purple)


kimp=remove_non_listed_pixels(keratin_binary_color,grey_list)
dimp=remove_non_listed_pixels(dermis_bin_color,green_list)
edimp=remove_non_listed_pixels(epidermis_binary_color,purple_list)
jimp=remove_non_listed_pixels(jun_bin_color,pink_list)


mer=merge_images((kimp,jimp,dimp,edimp))
cv2.imshow('1st',mer)

cv2.waitKey()

dimp=replace_black_spots(dimp,green)
kimp=replace_black_spots(kimp,grey)
edimp=replace_black_spots(edimp,purple)
jimp=replace_black_spots(jimp,pink)



mer=merge_images((kimp,jimp,dimp,edimp))
cv2.imshow(' imp',mer)

cv2.waitKey()


dice=dice_cof(mask,mer)
print("\n\n\n\n Dice coefficient:")
print(dice)


print('\n\n\n...removing BG now....\n\n\n')

cv2.imshow('bg Removed',rem)
cv2.waitKey()