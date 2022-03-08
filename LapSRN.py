import tensorflow as tf
import math
import numpy as np
import itertools
from VGGLoss import VGGLoss

class LapSRN:
    def __init__(self, input, scale, batch_size, learning_rate, vgg_layer, input_shape=(128, 128)):
        self.LR_input = input   # Low resolution inputs
        self.batch_size = batch_size
        self.scale = int(scale)
        self.num_of_components = int(math.floor(math.log(self.scale, 2)))
        self.learning_rate = learning_rate
        self.saver = ""
        self.filter_initializer = tf.keras.initializers.GlorotNormal()
        self.bias_initializer = tf.constant_initializer(value=0.1)
        self.outputs = list()
        self.global_step = tf.compat.v1.placeholder(tf.int32, shape=[], name="global_step")
        self.loss = VGGLoss(input_shape, vgg_layer)

    def activation(self, layer):
        return tf.nn.relu(layer) - 0.2 * tf.nn.relu(-1 * layer)

    def conv_layer(self, input_layer, filter, bias):
        conv_layer = tf.nn.conv2d(input=input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        conv_layer = self.activation(conv_layer + bias)
        return conv_layer

    def subpixel(self, X: tf.Tensor, upscale_factor):
        # Implementation of subpixel layer provided on https://neuralpixels.com/subpixel-upscaling/
        batch_size, rows, cols, in_channels = X.get_shape().as_list()
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
        new_shape = [batch_size, new_rows, new_cols, out_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, upscale_factor, upscale_factor, 1]
        out = tf.nn.conv2d_transpose(X, kernel, tf_shape, strides_shape, padding='VALID')
        return out

    def deconv_layer_pixel_shuffle(self, input_layer, channel_number):
        """Layer used for upsampling the image"""
        current_scale = 2
        filter_name = 'reconstruction_' + str(current_scale) + "_deconv_f"
        filter = tf.Variable(initial_value=self.filter_initializer(shape=(4, 4, channel_number,
                                                                          (current_scale * current_scale))),
                             name=filter_name)

        deconv_layer = tf.nn.conv2d(input=input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME',
                                    data_format='NHWC')
        deconv_layer = tf.nn.depth_to_space(deconv_layer, current_scale, data_format='NHWC')

        return deconv_layer

    def feature_upsample_layer(self, input_layer, scale):
        # layer_upsampled = self.deconv_layer_pixel_shuffle(input_layer, channel_number)
        layer_upsampled = self.subpixel(input_layer, scale)
        return layer_upsampled

    def image_upsample_layer(self, input_layer, channel):
        layer_upsampled = self.deconv_layer_pixel_shuffle(input_layer, channel)
        return layer_upsampled

    def feature_extraction(self, input_layer, index):
        """
        Feature extraction subnetwork. It is made of a cascade of convolutional layers followed by the upsampling layer
        """
        filter_size = 3
        filter_0 = tf.Variable(initial_value=self.filter_initializer(shape=(filter_size, filter_size, 1, 64)),
                               name="0_" + str(index) + "f")
        bias_0 = tf.compat.v1.get_variable(shape=[1], initializer=self.bias_initializer, name="0_" + str(index) + "bias")     # get the variable with the specified name
        layer_fe = self.conv_layer(input_layer, filter_0, bias_0)
        for i in range(1, 10):
            filter = tf.Variable(initial_value=self.filter_initializer(shape=(filter_size, filter_size, 64, 64)))
            bias = tf.compat.v1.get_variable(shape=[64], initializer=self.bias_initializer,
                                   name=str(i) + "_" + str(index) + "bias")
            layer_fe = self.conv_layer(layer_fe, filter, bias)

        layer_fe = self.feature_upsample_layer(layer_fe, 2)
        return self.activation(layer_fe)

    def LapSRN_model(self):
        """
        Implementation of LapSRN

        Returns
        ----------
        Model
        """
        outputs = list()
        prev_fe_layer = self.LR_input
        prev_re_layer = self.LR_input

        for n in range(0, self.num_of_components):
            current_scale = pow(2, (n + 1))
            fe_output = self.feature_extraction(prev_fe_layer, n)
            upsampled_image = self.image_upsample_layer(prev_re_layer, 1)
            re_output = self.activation(tf.math.add(upsampled_image, fe_output))

            prev_fe_layer = fe_output
            prev_re_layer = re_output
            outputs.append(re_output)

            output_name = "NCHW_output_" + str(current_scale) + "x"
            tf.transpose(re_output, [0, 3, 1, 2], name=output_name)

        return outputs

    def LapSRN_trainable_model_multi(self, HR_outputs, HR_origs):
        losses = list()
        train_ops = list()
        psnrs = list()

        for n in range(0, len(HR_outputs)):
            psnr = tf.image.psnr(HR_outputs[n], HR_origs[n], max_val=1.0)

            loss = self.loss.compute_loss(HR_outputs[n], HR_origs[n])


            decayed_lr = tf.train.exponential_decay(self.learning_rate,
                                                    self.global_step, 10000,
                                                    0.95, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(loss)

            losses.append(loss)
            train_ops.append(train_op)
            psnrs.append(psnr)

        return losses, train_ops, psnrs

    def LapSRN_trainable_model(self, HR_out, HR_orig):
        psnr = tf.image.psnr(HR_out, HR_orig, max_val=1.0)

        loss = self.loss.compute_loss(HR_out, HR_orig)
        decayed_lr = tf.train.exponential_decay(self.learning_rate,
                                                self.global_step, 10000,
                                                0.95, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(loss)

        return loss, train_op, psnr

