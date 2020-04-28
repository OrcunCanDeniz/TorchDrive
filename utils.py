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


def choose_image(imgQuery, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return preprocess(load_image(imgQuery['left'])), steering_angle + 0.2
    elif choice == 1:
        return preprocess(load_image(imgQuery['right'])), steering_angle - 0.2
    return preprocess(load_image(imgQuery['center'])), steering_angle


class Transforms(object):
    def random_flip(self, image, steering_angle):
        """
        Randomly flip the image left <-> right, and adjust the steering angle.
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def random_translate(self, image, steering_angle, range_x, range_y):
        """
        Randomly shift the image vertically and horizontally (translation).
        """
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    def random_shadow(self, image):
        """
        Generates and adds random shadow
        """

        x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
        xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(self, image):
        """
        Randomly adjust brightness of the image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def augument(self, sample, range_x=100, range_y=10):
        """
        Generate an augumented image and adjust steering angle.
        """
        image, steering_angle = sample[0], sample[1]
        image, steering_angle = self.random_flip(image, steering_angle)
        image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)
        image = self.random_shadow(image)
        image = self.random_brightness(image)
        return (image, steering_angle)


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
