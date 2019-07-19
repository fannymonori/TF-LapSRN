import tensorflow as tf
import pathlib
import cv2
import numpy as np
import os
import logging
import sys
import math

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

# from tensorflow.python import debug as tf_debug

import LapSRN
import utils

def create_dataset_generator(image_paths, scale):
    if scale == 2:
        return tf.data.Dataset.from_generator(
            utils.gen_dataset_multiscale, (tf.float32, tf.float32),
            (tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
            args=[image_paths, scale])
    elif scale == 4:
        return tf.data.Dataset.from_generator(
            utils.gen_dataset_multiscale, (tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
            args=[image_paths, scale])
    elif scale == 8:
        return tf.data.Dataset.from_generator(
            utils.gen_dataset_multiscale, (tf.float32, tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1]),
             tf.TensorShape([None, None, 1])),
            args=[image_paths, scale])


def get_model(scale, batch, lrate, iter):
    if scale == 2:
        LR, HR = iter.get_next()
        M = LapSRN.LapSRN(input=LR, scale=scale, batch_size=batch, learning_rate=lrate)
        outputs = M.LapSRN_model()
        loss, train_op, psnr = M.LapSRN_trainable_model(outputs[0], HR)
        return loss, train_op, psnr, M
    if scale == 4:
        LR, HR_1, HR_2 = iter.get_next()
        M = LapSRN.LapSRN(input=LR, scale=scale, batch_size=batch, learning_rate=lrate)
        outputs = M.LapSRN_model()
        loss, train_op, psnr = M.LapSRN_trainable_model_multi(outputs, [HR_1, HR_2])
        return loss, train_op, psnr, M
    if scale == 8:
        LR, HR_1, HR_2, HR_3 = iter.get_next()
        M = LapSRN.LapSRN(input=LR, scale=scale, batch_size=batch, learning_rate=lrate)
        outputs = M.LapSRN_model()
        loss, train_op, psnr = M.LapSRN_trainable_model_multi(outputs, [HR_1, HR_2, HR_3])
        return loss, train_op, psnr, M

def training(ARGS):
    """
    Start training the LapSRN model.
    """

    print("\nStarting training...\n")

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    SCALE = ARGS["SCALE"]
    BATCH = 32
    EPOCHS = ARGS["EPOCH_NUM"]
    DATA = pathlib.Path(ARGS["TRAINDIR"])
    LRATE = ARGS["LRATE"]

    all_image_paths = list(DATA.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ds = create_dataset_generator(all_image_paths, SCALE)

    train_dataset = ds.batch(BATCH)
    train_dataset = train_dataset.shuffle(10000)
    iter = train_dataset.make_initializable_iterator()

    loss, train_op, psnr, M = get_model(SCALE, BATCH, LRATE, iter)

    with tf.Session(config=config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

        saver = M.saver
        if not os.path.exists(ARGS["CKPT_dir"]):
            os.makedirs(ARGS["CKPT_dir"])
        else:
            if os.path.isfile(ARGS["CKPT"] + ".meta"):
                saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
                print("Loaded checkpoint.")
            else:
                print("Previous checkpoint does not exists.")

        # training with tf.data method
        saver = tf.train.Saver()

        if SCALE > 2:
            train_args = loss + train_op + psnr
        else:
            train_args = [loss, train_op, psnr]

        for e in range(EPOCHS):
            sess.run(iter.initializer)
            count = 0

            r = int(math.log(SCALE, 2))
            print(r)
            train_loss = np.zeros(r, dtype=float)
            train_psnr = np.zeros(r, dtype=float)

            while True:
                try:
                    count = count + 1

                    lt = list(sess.run(train_args, feed_dict={M.global_step: count}))

                    if SCALE == 2:
                        l = lt[0]
                        ps = lt[2]
                    elif SCALE == 4:
                        l = lt[0:2]
                        ps = lt[4:6]
                    elif SCALE == 8:
                        l = lt[0:3]
                        ps = lt[6:9]

                    train_loss += l

                    if SCALE == 2:
                        train_psnr += (np.mean(np.asarray(np.asarray(ps)), axis=0))
                    else:
                        train_psnr += (np.mean(np.asarray(ps), axis=1))

                    if count % 100 == 0:
                        logging.info('Epoch no: %04d' % (e + 1))
                        for n in range(0, r):
                            log_msg = "Output #%d Loss: %s Epoch loss: %s Epoch PSNR: %s"
                            if SCALE > 2:
                                logging.info(log_msg, n, "{:.9f}".format(l[n]),
                                             "{:.9f}".format(train_loss[n] / (count)),
                                             "{:.9f}".format(train_psnr[n] / (count)))
                            elif SCALE == 2:
                                logging.info(log_msg, n, "{:.9f}".format(l),
                                             "{:.9f}".format(train_loss[0] / (count)),
                                             "{:.9f}".format(train_psnr[0] / (count)))

                        saver.save(sess, ARGS["CKPT"])

                except tf.errors.OutOfRangeError:
                    break

            saver.save(sess, ARGS["CKPT"])

        train_writer.close()


def test(ARGS):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    SCALE = ARGS['SCALE']

    path = ARGS["TESTIMG"]

    fullimg = cv2.imread(path, 3)
    width = fullimg.shape[0]
    height = fullimg.shape[1]

    cropped = fullimg[0:(width - (width % SCALE)), 0:(height - (height % SCALE)), :]
    img = cv2.resize(cropped, None, fx=1. / SCALE, fy=1. / SCALE, interpolation=cv2.INTER_CUBIC)
    floatimg = img.astype(np.float32) / 255.0

    # Convert to YCbCr color space
    imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_BGR2YCrCb)
    imgY = imgYCbCr[:, :, 0]

    LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)

    with tf.Session(config=config) as sess:
        print("\nStart running tests on the model\n")
        # #load the model with tf.data generator
        ckpt_name = ARGS["CKPT"] + ".meta"
        saver = tf.train.import_meta_graph(ckpt_name)
        saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
        graph_def = sess.graph

        LR_tensor = sess.graph.get_tensor_by_name("IteratorGetNext:0")
        inp = cv2.cvtColor((cropped.astype(np.float32) / 255.0), cv2.COLOR_BGR2YCrCb)[:, :, 0].reshape(1,
                                                                                                       cropped.shape[0],
                                                                                                       cropped.shape[1],
                                                                                                       1)
        bicub = cv2.cvtColor(cv2.resize(cropped, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC),
                             cv2.COLOR_BGR2YCrCb)

        output = sess.run(sess.graph.get_tensor_by_name("NCHW_output:0"), feed_dict={LR_tensor: inp})

        Y = output[0][0]

        cv2.imshow('LapSRN HR image', Y)
        cv2.imshow('Bicubic HR image', bicub[:, :, 0])
        cv2.waitKey(0)

        LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")

        HR_tensor = graph_def.get_tensor_by_name("NCHW_output:0")

        output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

        Y = output[0][0]
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        Cr = np.expand_dims(cv2.resize(imgYCbCr[:, :, 1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCbCr[:, :, 2], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC),
                            axis=2)

        HR_image_YCrCb = np.concatenate((Y, Cr, Cb), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2BGR)) * 255.0).clip(min=0, max=255)
        HR_image = (HR_image).astype(np.uint8)

        bicubic_image = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

        print(np.amax(Y), np.amax(LR_input_))
        print("PSNR of LapSRN generated image: ", utils.PSNR(cropped, HR_image))
        print("PSNR of bicubic interpolated image: ", utils.PSNR(cropped, bicubic_image))

        cv2.imshow('Original image', fullimg)
        cv2.imshow('HR image', HR_image)
        cv2.imshow('Bicubic HR image', bicubic_image)
        cv2.waitKey(0)


def export(ARGS):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("\nStart exporting the model...\n")
    with tf.Session(config=config) as sess:
        ckpt_name = ARGS["CKPT"] + ".meta"
        saver = tf.train.import_meta_graph(ckpt_name)
        saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))

        # SAVE NCHW
        graph_def = sess.graph.as_graph_def()

        inputs = ['IteratorGetNext']

        if ARGS['SCALE'] == 4:
            outputs = ['NCHW_output', 'NCHW_output_0']
        elif ARGS['SCALE'] == 8:
            outputs = ['NCHW_output', 'NCHW_output_0', 'NCHW_output_1']
        else:
            outputs = ['NCHW_output']
        type = ["DT_FLOAT"]
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, outputs)
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, inputs, outputs, type)
        graph_def = TransformGraph(graph_def, inputs, outputs, ['sort_by_execution_order'])

        filename = "export/LapSRN_" + str(ARGS['SCALE']) + ".pb"
        with tf.gfile.FastGFile(filename, 'wb') as f:
            f.write(graph_def.SerializeToString())

        # tf.train.write_graph(graph_def, "", 'export/stripped.pbtxt', as_text=True)

    print("\nExporting done!\n")
