import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(imgQuery, telemetry):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        img =  preprocess(load_image(imgQuery['left']))
        telemetry["steering"] += 0.2
    elif choice == 1:
        img =  preprocess(load_image(imgQuery['right']))
        telemetry["steering"] -= 0.2
    else:
        img = preprocess(load_image(imgQuery['center']))

    return [img,float(imgQuery["speed"]),np.array(telemetry)]

class Transforms(object):
    def random_flip(self, data):
        """
        Randomly flip the image left <-> right, and adjust the steering angle.
        """

        if np.random.rand() < 0.5:
            data[0] = cv2.flip(data[0], 1)
            data[2][0] = -data[2][0]
        return data

    def random_translate(self, data, range_x, range_y):
        """
        Randomly shift the image vertically and horizontally (translation).
        """


        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        data[2][0] += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = data[0].shape[:2]
        data[0] = cv2.warpAffine(data[0], trans_m, (width, height))
        return data

    def random_shadow(self, data):
        """
        Generates and adds random shadow
        """

        x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
        xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

        mask = np.zeros_like(data[0][:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(data[0], cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        data[0] = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        return data

    def random_brightness(self, data):
        """
        Randomly adjust brightness of the image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(data[0], cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        data[0] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return data

    def augument(self, sample, range_x=100, range_y=10):
        """
        Generate an augumented image and adjust steering angle.
        """

        sample = self.random_flip(sample)
        sample = self.random_translate(sample, range_x, range_y)
        sample = self.random_shadow(sample)
        sample = self.random_brightness(sample)
        return sample


def vis_train(trainingMetric, validationMetric, epochs, metricType):
    fig = plt.figure()
    x_vals = np.linspace(0, epochs-1, epochs, dtype=np.int)
    plt.plot(x_vals,trainingMetric, 'b', x_vals, validationMetric, 'g')
    plt.ylabel(metricType)
    plt.xlabel('Epochs')
    plt.title(metricType + ' over Epochs')
    plt.xlim(0, epochs-1)
    blue_patch = mpatches.Patch(color='blue', label='Training '+metricType)
    green_patch = mpatches.Patch(color='green', label='Validation '+metricType)
    plt.legend(handles=[blue_patch, green_patch])
    plt.show()
    figName = f'self_driving_car_{metricType}.png'
    fig.savefig(figName)
