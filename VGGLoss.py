from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Concatenate
import tensorflow as tf


class VGGLoss:
    def __init__(self, feat_extraction_layer):
        self.vgg = VGG19(include_top=False, input_shape=(256, 256, 3))
        self.model = Model(inputs=self.vgg.inputs, outputs=self.vgg.get_layer(feat_extraction_layer).output)

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
        img_predicted_tens = Concatenate()([predicted_image, predicted_image, predicted_image])
        img_gt_tens = Concatenate()([ground_truth_image, ground_truth_image, ground_truth_image])
        img_lr_feats = self.model(img_predicted_tens)
        img_hr_feats = self.model(img_gt_tens)
        c = tf.reduce_mean(tf.subtract(img_hr_feats, img_lr_feats)**2, [1, 2, 3])
        return tf.reduce_mean(tf.subtract(img_hr_feats, img_lr_feats)**2, [1, 2, 3])
