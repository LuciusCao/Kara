import numpy as np
import tensorflow as tf


def mu_encoder(audio, quantization_channels):
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        magnitute = tf.log1p(mu * tf.abs(audio)) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitute
        transformed_audio = tf.to_int32((signal + 1) / 2 * mu + 0.5)
        return transformed_audio


def mu_decoder(transformed_audio, quantization_channels):
    with tf.name_scope('decode'):
        mu = tf.to_float(quantization_channels - 1)
        signal = 2 * (tf.to_float(transformed_audio) / mu) - 1
        magnitute = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
        audio = tf.sign(signal) * magnitute
        return audio


def time_to_batch(value, dilation):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation):
    with tf.name_scope('causal_conv'):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
        return result
