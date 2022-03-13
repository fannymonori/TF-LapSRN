import tensorflow as tf
import math
import numpy as np
import itertools
from VGGLoss import VGGLoss
from keras.layers import Conv2DTranspose, Conv2D, LeakyReLU, Input
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.optimizers import schedules

class LapSRN:
    def __init__(self, scale, learning_rate, alpha=0.2, vgg_layer='block5_conv4', input_shape=(128, 128)):
        self.input_shape = input_shape
        self.scale = int(scale)
        self.num_of_components = int(math.floor(math.log(self.scale, 2)))
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.filter_initializer = tf.keras.initializers.GlorotNormal()
        self.bias_initializer = tf.constant_initializer(value=0.1)
        self.outputs = list()
        self.loss = VGGLoss(vgg_layer)

    def activation(self, layer):
        return tf.nn.relu(layer) - 0.2 * tf.nn.relu(-1 * layer)

    def subpixel(self, X: tf.Tensor, upscale_factor):
        # Implementation of subpixel layer provided on https://neuralpixels.com/subpixel-upscaling/
        shape = X.get_shape().as_list()
        batch, rows, cols, in_channels = [1, shape[1], shape[2], shape[3]]
        kernel_filter_size = upscale_factor
        out_channels = int(in_channels // (upscale_factor * upscale_factor))

        kernel_shape = [kernel_filter_size, kernel_filter_size, out_channels, in_channels]
        kernel = np.zeros(kernel_shape, np.float32)

        # Build the kernel so that a 4 pixel cluster has each pixel come from a separate channel.
        for c in range(0, out_channels):
            i = 0
            for x, y in itertools.product(range(upscale_factor), repeat=2):
                kernel[y, x, c, c * upscale_factor * upscale_factor + i] = 1
                i += 1

        new_rows, new_cols = int(rows * upscale_factor), int(cols * upscale_factor)
        new_shape = [batch, new_rows, new_cols, out_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, upscale_factor, upscale_factor, 1]
        out = tf.nn.conv2d_transpose(X, kernel, tf_shape, strides_shape, padding='VALID')
        out = tf.expand_dims(tf.reduce_mean(out, axis=-1), axis=-1)
        return out

    def deconv_layer_pixel_shuffle(self, input_layer, channels):
        """Layer used for upsampling the image"""
        upscale_factor = 2
        deconv_layer = Conv2D(filters=upscale_factor*upscale_factor*channels, kernel_size=4, strides=(1, 1), padding='SAME')(input_layer)
        deconv_layer = tf.nn.depth_to_space(deconv_layer, upscale_factor, data_format='NHWC')
        return deconv_layer

    def feature_upsample_layer(self, input_layer, scale):
        layer_upsampled = self.subpixel(input_layer, scale)
        return layer_upsampled

    def image_upsample_layer(self, input_layer, channel):
        layer_upsampled = self.deconv_layer_pixel_shuffle(input_layer, channel)
        return layer_upsampled

    def feature_extraction_block(self, input_layer):
        """
        Feature extraction subnetwork. It is made of a cascade of convolutional layers followed by the upsampling layer
        """
        layer_fe = LeakyReLU(alpha=self.alpha)(input_layer)
        for i in range(1, 10):
            layer_fe = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='SAME')(layer_fe)
            layer_fe = LeakyReLU(alpha=self.alpha)(layer_fe)
        layer_fe += input_layer     # Residual connection
        layer_fe = self.feature_upsample_layer(layer_fe, 2)
        return layer_fe

    def LapSRN_model(self):
        input_layer = Input((self.input_shape[0], self.input_shape[1], 1))
        prev_fe_layer = input_layer
        prev_re_layer = input_layer
        for n in range(0, self.num_of_components):
            fe_output = self.feature_extraction_block(prev_fe_layer)
            upsampled_image = self.image_upsample_layer(prev_re_layer, 1)
            re_output = tf.math.add(upsampled_image, fe_output)
            prev_fe_layer = fe_output
            prev_re_layer = re_output
        m = Model(input_layer, prev_re_layer)
        return m

    def prepare_model(self):
        def psnr(im1, im2):
            return tf.image.psnr(im1, im2, max_val=3.0)

        model = self.LapSRN_model()
        schedule = schedules.ExponentialDecay(self.learning_rate, 10000, 0.95, staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=schedule)
        model.compile(optimizer=opt, loss=self.loss.compute_loss, metrics=[psnr], run_eagerly=True)
        return model

    '''def LapSRN_trainable_model_multi(self, HR_outputs, HR_origs):
        losses = list()
        train_ops = list()
        psnrs = list()

        for n in range(0, len(HR_outputs)):
            psnr = tf.image.psnr(HR_outputs[n], HR_origs[n], max_val=1.0)

            loss = self.loss.compute_loss(HR_outputs[n], HR_origs[n])
            decayed_lr = tf.keras.optimizers.schedule.ExponentialDecay(self.learning_rate, 10000, 0.95, staircase=True)
            # decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.95, staircase=True)
            train_op = tf.keras.optimizers.Adam(learning_rate=decayed_lr).minimize(loss)

            losses.append(loss)
            train_ops.append(train_op)
            psnrs.append(psnr)

        return losses, train_ops, psnrs

    def LapSRN_trainable_model(self, HR_out, HR_orig):
        psnr = tf.image.psnr(HR_out, HR_orig, max_val=1.0)

        loss = self.loss.compute_loss(HR_out, HR_orig)
        decay_schedule = tf.keras.optimizers.schedule.ExponentialDecay(self.learning_rate, 10000, 0.95, staircase=True)
        # decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.95, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=decay_schedule)

        return loss, train_op, psnr'''

'''
from LapSRN import LapSRN
import numpy as np
from tensorflow.keras.optimizers import schedules
import tensorflow as tf

lr_im = "raw_dataset/train/lr_x2/train/lr_10_0.npy"
hr_im = "raw_dataset/train/gt/x2_train/hr_10_0.npy"
with open(lr_im, 'rb') as f:
    im = np.load(f)

with open(hr_im, 'rb') as f:
    im_hr = np.load(f)

c = LapSRN(im, 2, 1, 0.007, 2)
decay_schedule = schedules.ExponentialDecay(c.learning_rate, 10000, 0.95, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=decay_schedule)
m = c.prepare_model()
im = np.expand_dims(im, axis=(0, -1))
im_hr = np.expand_dims(im_hr, axis=(0, -1))
m.fit(im, im_hr, batch_size=1, epochs=1)
'''
