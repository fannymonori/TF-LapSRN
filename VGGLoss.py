from keras.applications.vgg19 import VGG19
from keras.models import Model
import tensorflow as tf


class VGGLoss:
    def __init__(self, input_shape, feat_extraction_layer='block5_conv4'):
        self.vgg = VGG19(include_top=False, input_shape=input_shape)
        self.model = Model(inputs=self.vgg.inputs, outputs=self.vgg.get_layer(feat_extraction_layer).outputs)

    def compute_loss(self, predicted_image, ground_truth_image):
        """
        Compute the perceptual loss between two images by calculating the mean squared error between the feature maps
        that the VGG19 computed for each of the two images at a specific layer. Since the images are single channel and
        the VGG expects 3-channel images, they are repeated thrice along the axis 2 in order to create a suitable input
        for the network
        :param predicted_image:
        :param ground_truth_image:
        :return:
        """
        img_lr_tens = tf.constant([predicted_image, predicted_image, predicted_image])
        img_lr_tens = tf.reshape(img_lr_tens, (1, img_lr_tens.shape[1], img_lr_tens.shape[2], img_lr_tens.shape[0]))

        img_hr_tens = tf.constant([ground_truth_image, ground_truth_image, ground_truth_image])
        img_hr_tens = tf.reshape(img_hr_tens, (1, img_hr_tens.shape[1], img_hr_tens.shape[2], img_hr_tens.shape[0]))

        img_lr_feats = self.model(img_lr_tens)
        img_hr_feats = self.model(img_hr_tens)

        return tf.reduce_mean(tf.subtract(img_hr_feats, img_lr_feats)**2, [1, 2, 3]).numpy[0]
