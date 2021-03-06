# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import warnings

from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D

from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

from .inputs import depth_input, rgb_input
from .outputs import classification_output, regression_output, biternion_output

from .inputs import INPUT_TYPES
from .inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from .outputs import OUTPUT_TYPES
from .outputs import OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION, OUTPUT_BITERNION


INPUT_SHAPES = ((123, 54), (68, 68), (46, 46))


def conv_bn_act(x, name_prefix,
                n_filters, kernel_size, strides, kernel_regularizer=None,
                padding='valid',
                sampling=False,
                activation='relu'):
    # conv
    x = Conv2D(n_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=True,
               kernel_regularizer=kernel_regularizer,
               name=name_prefix+'_conv')(x)
    # bn
    channel_axis = 3 if K.image_data_format() == 'channels_last' else 1
    use_train_mode = False if sampling else None   # None -> K.learning_phase()
    x = BatchNormalization(axis=channel_axis,
                           name=name_prefix+'_bn')(x, training=use_train_mode)
    # act
    x = Activation(activation=activation,
                   name=name_prefix+'_act')(x)

    return x


def get_model(input_type=INPUT_DEPTH,
              input_shape=(46, 46),
              output_type=OUTPUT_BITERNION,
              n_classes=None,
              weight_decay=None,
              activation='relu',
              sampling=False,
              **kwargs):
    # check arguments
    assert weight_decay is None    # not used in original implementation
    assert input_type in INPUT_TYPES
    assert input_shape in INPUT_SHAPES
    assert output_type in OUTPUT_TYPES
    assert n_classes is not None or output_type != OUTPUT_CLASSIFICATION

    for kw in kwargs:
        warnings.warn("argument '{}' not supported for Beyer".format(kw))

    # regularizer
    reg = l2(weight_decay) if weight_decay is not None else None

    # define inputs -----------------------------------------------------------
    inputs = []
    if input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        inputs.append(depth_input(input_shape))
    if input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        inputs.append(rgb_input(input_shape))

    # input stage (apply same structure to all inputs) ------------------------
    input_stage_outputs = []
    for inp in inputs:
        # extract name (input of type tensorflow.Tensor)
        inp_name = inp.name.split(':')[0]

        x = inp
        x = conv_bn_act(x, name_prefix='{}_1'.format(inp_name),
                        n_filters=24, kernel_size=(3, 3), strides=(1, 1),
                        kernel_regularizer=reg, padding='valid',
                        sampling=sampling,
                        activation=activation)
        x = conv_bn_act(x, name_prefix='{}_2'.format(inp_name),
                        n_filters=24, kernel_size=(3, 3), strides=(1, 1),
                        kernel_regularizer=reg, padding='valid',
                        sampling=sampling,
                        activation=activation)
        if input_shape == (123, 54):
            # append another block
            x = conv_bn_act(x, name_prefix='{}_3'.format(inp_name),
                            n_filters=24, kernel_size=(3, 3), strides=(1, 1),
                            kernel_regularizer=reg, padding='valid',
                            sampling=sampling,
                            activation=activation)
        input_stage_outputs.append(x)

    # fuse input stages
    if len(input_stage_outputs) > 1:
        # multiple sages
        # concatenate
        concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = concatenate(input_stage_outputs, axis=concat_axis,
                        name='input_fusion_concat')
        # apply 1x1 to reduce the number of channels to 24
        x = conv_bn_act(x, name_prefix='input_fusion',
                        n_filters=24, kernel_size=(1, 1), strides=(1, 1),
                        kernel_regularizer=reg, padding='valid',
                        sampling=sampling,
                        activation=activation)
    else:
        # only one stage
        x = input_stage_outputs[0]

    # main stage --------------------------------------------------------------
    if input_shape == (123, 54):
        # apply 3x2 non-overlapping pooling
        x = MaxPool2D(pool_size=(3, 2), strides=(3, 2), name='main_1_pool')(x)
    else:
        # apply 2x2 default non-overlapping pooling
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='main_1_pool')(x)

    x = conv_bn_act(x, name_prefix='main_2',
                    n_filters=48, kernel_size=(3, 3), strides=(1, 1),
                    kernel_regularizer=reg, padding='valid',
                    sampling=sampling,
                    activation=activation)
    x = conv_bn_act(x,  name_prefix='main_3',
                    n_filters=48, kernel_size=(3, 3), strides=(1, 1),
                    kernel_regularizer=reg, padding='valid',
                    sampling=sampling,
                    activation=activation)

    if input_shape == (123, 54):
        # append another block
        x = conv_bn_act(x,  name_prefix='main_4',
                        n_filters=48, kernel_size=(3, 3), strides=(1, 1),
                        kernel_regularizer=reg, padding='valid',
                        sampling=sampling,
                        activation=activation)

    if input_shape == (123, 54):
        # apply 3x3 non-overlapping pooling
        x = MaxPool2D(pool_size=(3, 3), strides=(3, 3), name='main_5_pool')(x)
    else:
        if input_shape == (46, 46):
            # add padding to get a 9x9 output, otherwise the output is 8x8
            # see: https://github.com/lucasb-eyer/BiternionNet/blob/master/
            #           Experiments%20-%20TownCentre.ipynb
            if activation != 'relu':
                warnings.warn("Since `activation` is not 'relu', zero padding "
                              "may affect the results")
            x = ZeroPadding2D(padding=((0, 1), (0, 1)),
                              name='main_5_pad')(x)

        # apply 2x2 default non-overlapping pooling
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='main_5_pool')(x)

    x = conv_bn_act(x, name_prefix='main_6',
                    n_filters=64, kernel_size=(3, 3), strides=(1, 1),
                    kernel_regularizer=reg, padding='valid',
                    sampling=sampling,
                    activation=activation)
    x = conv_bn_act(x, name_prefix='main_7',
                    n_filters=64, kernel_size=(3, 3), strides=(1, 1),
                    kernel_regularizer=reg, padding='valid',
                    sampling=sampling,
                    activation=activation)

    if input_shape == (68, 68):
        # apply another 2x2 non-overlapping pooling
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='main_8_pool')(x)

    # output stage ------------------------------------------------------------
    use_train_mode = True if sampling else None    # None -> K.learning_phase()
    x = Flatten(name='output_1_flatten')(x)
    x = Dropout(rate=0.2, name='output_2_dropout')(x, training=use_train_mode)
    x = Dense(units=512, kernel_regularizer=reg, name='output_2_dense')(x)
    x = Activation(activation, name='output_2_act')(x)
    x = Dropout(rate=0.5, name='output_3_dropout')(x, training=use_train_mode)

    if output_type == OUTPUT_BITERNION:
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.01)
        x = biternion_output(kernel_initializer=kernel_initializer,
                             kernel_regularizer=reg,
                             name='output_3_dense_and_act')(x)
    elif output_type == OUTPUT_REGRESSION:
        x = regression_output(kernel_regularizer=reg,
                              name='output_3_dense_and_act')(x)
    elif output_type == OUTPUT_CLASSIFICATION:
        x = classification_output(n_classes, name='output_3_dense_and_act',
                                  kernel_regularizer=reg)(x)

    return Model(inputs=inputs, outputs=[x])
