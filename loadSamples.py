import gzip
import numpy as np

def loadImages(mode):
    if mode == 'train':
        f_images = gzip.open('./mnist/train-images-idx3-ubyte.gz', 'r')
        totalImages = 60000
    elif mode == 'test':
        f_images = gzip.open('./mnist/t10k-images-idx3-ubyte.gz', 'r')
        totalImages = 10000
    f_images.seek(0, 0)
    f_images.seek(16)
    c = 0
    images = np.ndarray(shape=(totalImages, 784))
    while c < totalImages:
        image_buf = f_images.read(784) 
        image = np.frombuffer(image_buf, dtype=np.uint8).astype(np.float32)
        images[c] = image
        c+=1
    return images

def loadLabels(mode):
    
    if mode == 'train':
        f_labels = gzip.open('./mnist/train-labels-idx1-ubyte.gz', 'r')
        totalLabels = 60000
    elif mode == 'test':
        f_labels = gzip.open('./mnist/t10k-labels-idx1-ubyte.gz', 'r')
        totalLabels = 10000
    f_labels.seek(0, 0)
    f_labels.seek(8)
    c = 0
    labels = np.ndarray(shape=(totalLabels, 1))
    while c < totalLabels:
        labels_buf = f_labels.read(1)
        label = np.frombuffer(labels_buf, dtype=np.uint8).astype(np.int64)
        labels[c] = label
        c+=1
    return labels

def selectImagesAndLabels(batch_size, images, labels):
    rng = np.random.default_rng() # so that the generated indices aren't chosen twice
    batchIdxs = rng.choice(images.shape[0], size=batch_size, replace=False)
    images_set = np.zeros(shape=(batch_size, 784))
    labels_set = np.zeros(shape=(batch_size, 1))
    for set_idx, idx in enumerate(batchIdxs):
        images_set[set_idx] = np.copy(images[idx])
        labels_set[set_idx] = np.copy(labels[idx])
    return images_set, labels_set
