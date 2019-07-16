import tensorflow as tf
import numpy as np
import math
import os


class LapSRN:

    def __init__(self, input, scale, batch_size, learning_rate):
        self.LR_input = input
        self.batch_size = batch_size
        self.scale = int(scale)
        self.num_of_components = int(math.floor(math.log(self.scale, 2)))
        self.learning_rate = learning_rate
        self.saver = ""
        #self.filters_fe = list()
        #self.biases_fe = list()

        # self.size = tf.placeholder(tf.int32, shape=[2], name="dimensions")
        self.size = tf.placeholder(tf.int32, shape=[4], name="dimensions")

        #self.filter_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.filter_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        self.bias_initializer = tf.constant_initializer(value=0.1)
        #self.deconv_bias = list()

        self.outputs = list()

        self.global_step = ""
        self.global_step = tf.placeholder(tf.int32, shape=[], name="global_step")


    def bilinear_filter(self, scalef, channels):
        size = 4

        bilinear_f = np.zeros([size, size, 1, channels])

        #factor = scalef
        factor = (size + 1) // 2

        c = factor - 0.5

        b = np.zeros((size, size))
        for x in range(0, size):
            for y in range(0, size):
                K = (1 - abs((x - c) / factor)) * (1 - abs((y - c) / factor))
                b[x, y] = K

        for i in range(0, channels):
            bilinear_f[:, :, 0, i] = b

        return tf.constant_initializer(value=bilinear_f, dtype=tf.float32)

    def activation(self, layer):
        #return tf.nn.leaky_relu(layer)
        return tf.nn.relu(layer)-0.2 * tf.nn.relu(-1*layer)
        #return tf.nn.relu(layer)


    def conv_layer(self, input_layer, filter, bias, name):
        filter_name = name + "conv2d"
        conv_layer = tf.nn.conv2d(input=input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        conv_layer = self.activation(conv_layer + bias)
        return conv_layer


    def deconv_layer_pixel_shuffle(self, input_layer, channel_number, index):

        current_scale = 2
        filter_name = 'reconstruction_' + str(current_scale) + "_deconv_f"
        filter = tf.Variable(initial_value=self.filter_initializer(shape=(4, 4, channel_number,
                                                                          (current_scale * current_scale))),name=filter_name)

        conv2d_name = 'reconstruction_' + str(current_scale) + 'conv2d'
        deconv_layer = tf.nn.conv2d(input=input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME',
                                data_format='NHWC')
        deconv_layer = tf.nn.depth_to_space(deconv_layer, current_scale, data_format='NHWC')

        return deconv_layer

    def upsample_layer(self, input_layer, channel_number, current_scale, index, name):
        layer_upsampled = self.deconv_layer_pixel_shuffle(input_layer, channel_number, index)
        #layer_upsampled = self.deconv_layer_transposed(input_layer, channel_number, current_scale, name)

        return layer_upsampled

    def deconv_layer_transposed(self, input_layer, channel_number, current_scale, name):

        filter_name = str(current_scale) + "_" + str(channel_number) + "transposed_filters_" + name
        filter = tf.get_variable(initializer=self.bilinear_filter(2, channel_number), shape=[4, 4, 1, channel_number],name=filter_name)
        #shape = tf.shape(input_layer)
        #size = [shape[0], shape[1] * 2, shape[2] * 2, 1]

        #w = int((128 / self.scale) * current_scale)
        #size = tf.Variable([w, w])
        #size = [shape[1] * 2, shape[2] * 2]
        #size = [32, w, w, 1]

        #deconv_layer = tf.nn.conv2d_transpose(value=input_layer, filter=filter, output_shape=size, strides=[1, 2, 2, 1], padding='SAME')

        deconv_layer = tf.layers.conv2d_transpose(input_layer, 1, [4, 4], [2, 2], "same", "channels_last", None, False, self.bilinear_filter(2, channel_number))

        #filter = tf.Variable(initial_value=self.filter_initializer(shape=(4, 4, channel_number, 1)),name=filter_name)
        #l = tf.image.resize_images(images=input_layer, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #deconv_layer = tf.nn.conv2d(l, filter, [1, 1, 1, 1], padding='SAME', name=name)

        return deconv_layer

    def deconv_layer_transposed_flip(self, input_layer, filter, channel_number, current_scale, name):
        transposed_weights = tf.transpose(filter, perm=[0, 1, 3, 2])
        transposed_weights = tf.reverse(transposed_weights, axis=[0, 1])
        tf.nn.conv2d(input_layer, transposed_weights, strides=[1, 1, 1, 1], padding='SAME')

    def feature_extraction(self, input_layer, current_scale, index):

        filter_size = 3

        filter_0 = tf.Variable(initial_value=self.filter_initializer(shape=(filter_size, filter_size, 1, 64)),
                    name="0_" + str(index) + "f")
        bias_0 = tf.get_variable(shape=[1], initializer=self.bias_initializer, name="0_" + str(index) + "bias")


        layer_fe = self.conv_layer(input_layer, filter_0, bias_0,
                                   name=str("input_") + str(index))
        for i in range(1, 10):
            filter = tf.Variable(initial_value=self.filter_initializer(shape=(filter_size, filter_size, 64, 64)))
            bias = tf.get_variable(shape=[64], initializer=self.bias_initializer, name=str(i) + "_" + str(index) + "bias")

            layer_fe = self.conv_layer(layer_fe, filter, bias, name=str(i) + "_" + str(index) + "conv2d")

        layer_fe = self.upsample_layer(layer_fe, 64, current_scale, index, "fe")

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
            # current_scale = 2
            #print "Current scale: ", current_scale

            fe_output = self.feature_extraction(prev_fe_layer, current_scale, n)

            layer_re = self.upsample_layer(prev_re_layer, 1, current_scale, n, "re")

            re_output = self.activation(tf.math.add(layer_re,fe_output))

            prev_fe_layer = fe_output
            prev_re_layer = re_output

            outputs.append(re_output)

            output_name = "NCHW_output_" + str(n)
            tf.transpose(re_output, [0, 3, 1, 2], name=output_name)

        out_nchw = tf.transpose(re_output, [0, 3, 1, 2], name="NCHW_output")

        self.saver = tf.train.Saver()

        # return re_output
        return outputs

    def LapSRN_trainable_model_multi(self, HR_outputs, HR_origs):

        losses = list()
        train_ops = list()
        psnrs = list()

        for n in range(0, len(HR_outputs)):
            psnr = tf.image.psnr(HR_outputs[n], HR_origs[n], max_val=1.0)

            # Charbonnier penalty function
            epsilon = 1e-6
            loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(HR_outputs[n], HR_origs[n])) + epsilon)))

            decayed_lr = tf.train.exponential_decay(self.learning_rate,
                                                    self.global_step, 10000,
                                                    0.95, staircase=True)
            #train_op = tf.contrib.opt.AdamWOptimizer(0.0001, learning_rate=self.learning_rate).minimize(loss)
            #train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            train_op = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(loss)

            losses.append(loss)
            train_ops.append(train_op)
            psnrs.append(psnr)

        return losses, train_ops, psnrs

    def LapSRN_trainable_model(self, HR_out, HR_orig):
        psnr = tf.image.psnr(HR_out, HR_orig, max_val=1.0)

        # Charbonnier penalty function
        epsilon = 1e-6
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(HR_out, HR_orig)) + epsilon)))

        # train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        train_op = tf.contrib.opt.AdamWOptimizer(0.0001, learning_rate=self.learning_rate).minimize(loss)

        return loss, train_op, psnr
