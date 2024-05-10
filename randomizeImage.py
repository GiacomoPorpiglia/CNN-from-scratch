from scipy.ndimage import rotate
from scipy.ndimage import zoom
import random
import numpy as np

#rotate by random angle between -10 and 10 degrees
def rotate_image(image):
    image = rotate(image, angle=random.randint(-10, 10), reshape=False)
    return image

#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def zoom_image(image, zoom_factor, **kwargs):
    h, w = image.shape[:2]

    zoom_tuple = (zoom_factor,) * 2 + (1,) * (image.ndim - 2)

    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(image)
        out[top:top+zh, left:left+zw] = zoom(image, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(image[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `image` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = image
    return out


def translate_image(image, translateX, translateY):

    if translateY > 0:
        for row_idx in range(image.shape[0]-1, translateY-1, -1):
            image[row_idx] = image[row_idx-translateY]
        for row_idx in range(translateY):
            image[row_idx] = np.zeros(image.shape[1])

    if translateY < 0:
        for row_idx in range(0, image.shape[0]+translateY, 1):
            image[row_idx] = image[row_idx - translateY]
        for row_idx in range(image.shape[0]-1, image.shape[0]-1+translateY, -1):
            image[row_idx] = np.zeros(image.shape[1])
    
    if translateX > 0:
        for row_idx in range(image.shape[0]):
            for col_idx in range(image.shape[1]-1, translateX-1, -1):
                image[row_idx, col_idx] = image[row_idx, col_idx - translateX]
            for col_idx in range(translateX):
                image[row_idx, col_idx] = 0

    if translateX < 0:
        for row_idx in range(image.shape[0]):
            for col_idx in range(0, image.shape[1]+ translateX, 1):
                image[row_idx, col_idx] = image[row_idx, col_idx-translateX]
            for col_idx in range(image.shape[1]-1, image.shape[1]-1+translateX, -1):
                image[row_idx, col_idx] = 0

    return image

def add_noise(image):
    for idx in range(len(image)):
        if random.random() < .03:
            image[idx] = max(image[idx], 0.7*random.random())
    return image

def clear_and_normalize(image):
    image /= 255
    image[image < .1] = 0
    return image
