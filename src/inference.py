# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
import seaborn
from scipy.stats import norm
from scipy.stats import circmean, circstd
import tensorflow.keras.backend as K

from deep_orientation import beyer    # noqa # pylint: disable=unused-import
from deep_orientation import mobilenet_v2    # noqa # pylint: disable=unused-import
from deep_orientation import beyer_mod_relu     # noqa # pylint: disable=unused-import

from deep_orientation.inputs import INPUT_TYPES
from deep_orientation.inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)
import deep_orientation.preprocessing as pre
import deep_orientation.postprocessing as post

import utils.img as img_utils
from utils.io import get_files_by_extension

# seaborn.set_style('darkgrid')
seaborn.set_context('notebook', font_scale=1.2)

import time

#----------- Crea la red neuronal ------------------------------
def load_network(model_name, weights_filepath,
                 input_type, input_height, input_width,
                 output_type,
                 sampling=False,
                 **kwargs):

    # load model --------------------------------------------------------------
    model_module = globals()[model_name]
    model_kwargs = {}
    if model_name == 'mobilenet_v2' and 'mobilenet_v2_alpha' in kwargs:
        model_kwargs['alpha'] = kwargs.get('mobilenet_v2_alpha')
    if output_type == OUTPUT_CLASSIFICATION:
        assert 'n_classes' in kwargs
        model_kwargs['n_classes'] = kwargs.get('n_classes')

    model = model_module.get_model(input_type=input_type,
                                   input_shape=(input_height, input_width),
                                   output_type=output_type,
                                   sampling=sampling,
                                   **model_kwargs)

    # load weights ------------------------------------------------------------
    model.load_weights(weights_filepath)

    return model


def main():
    #------------- Parametros necesarios para la ejecucion del script ------------------

    model_type = "mobilenet_v2"
    weights_filepath = "../trained_networks/mobilenet_v2_1_00__rgb__96x96__biternion__0_001000__0/weights_valid_0134.hdf5"
    image_or_image_basepath = "../nicr_rgb_d_orientation_data_set_examples/small_patches"
    input_type = "rgb"
    input_width = 96
    input_height = 96
    input_preprocessing = "scale01"
    n_samples = 1
    output_type = "biternion"
    n_classes = 8
    mobilenet_v2_alpha = 1.0
    devices = "0"
    cpu = True
    verbose = False
    #------------------------------------------------------------------------------------------
    # set device and data format ----------------------------------------------
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        devices = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    if not devices or model_type == 'mobilenet_v2':
        # note: tensorflow supports b01c pooling on cpu only
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    # load model --------------------------------------------------------------
    model = load_network(model_type, weights_filepath,
                         input_type, input_height, input_width,
                         output_type,
                         sampling=n_samples > 1,
                         n_classes=n_classes,
                         mobilenet_v2_alpha=mobilenet_v2_alpha)

    # parse for image files ---------------------------------------------------
    # note: we do not search for mask files, but derive masks from either the
    # depth or rgb image during preprocessing
    
    DEPTH_SUFFIX = '_Depth.pgm'
    RGB_SUFFIX = '_RGB.png'
    MASK_SUFFIX = '_Mask.png'
    # get filepaths
    mask_filepaths = get_files_by_extension(
            image_or_image_basepath, extension=MASK_SUFFIX.lower(),
            flat_structure=True, recursive=True, follow_links=True)

    if input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        depth_filepaths = get_files_by_extension(
            image_or_image_basepath, extension=DEPTH_SUFFIX.lower(),
            flat_structure=True, recursive=True, follow_links=True)
        assert len(depth_filepaths) == len(mask_filepaths)
        filepaths = list(zip(depth_filepaths, mask_filepaths))
        assert all(depth_fp.replace(DEPTH_SUFFIX, '') ==
                   mask_fp.replace(MASK_SUFFIX, '')
                   for depth_fp, mask_fp in filepaths)

    if input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        rgb_filepaths = get_files_by_extension(
            image_or_image_basepath, extension=RGB_SUFFIX.lower(),
            flat_structure=True, recursive=True, follow_links=True)
        assert len(rgb_filepaths) == len(mask_filepaths)
        filepaths = list(zip(rgb_filepaths, mask_filepaths))
        assert all(rgb_fp.replace(RGB_SUFFIX, '') ==
                   mask_fp.replace(MASK_SUFFIX, '')
                   for rgb_fp, mask_fp in filepaths)

    if input_type == INPUT_DEPTH_AND_RGB:
        filepaths = list(zip(depth_filepaths, rgb_filepaths, mask_filepaths))

    # define preprocessing function -------------------------------------------
    def load_and_preprocess(inputs):
        # unpack inputs
        if input_type == INPUT_DEPTH_AND_RGB:
            depth_filepath, rgb_filepath, mask_filepath = inputs
        elif input_type == INPUT_DEPTH:
            depth_filepath, mask_filepath = inputs
        else:
            rgb_filepath, mask_filepath = inputs

        # pack shape
        shape = (input_height, input_width)

        # load mask
        mask = img_utils.load(mask_filepath)
        mask_resized = pre.resize_mask(mask, shape)
        mask_resized = mask_resized > 0

        # prepare depth input
        if input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
            # load
            depth = img_utils.load(depth_filepath)

            # create mask
            # mask = depth > 0
            # mask_resized = pre.resize_mask(mask.astype('uint8')*255, shape) > 0

            # mask (redundant, since mask is derived from depth image)
            # depth = pre.mask_img(depth, mask)

            # resize
            depth = pre.resize_depth_img(depth, shape)

            # 01 -> 01c
            depth = depth[..., None]

            # preprocess
            depth = pre.preprocess_img(
                depth,
                mask=mask_resized,
                scale01=input_preprocessing == 'scale01',
                standardize=input_preprocessing == 'standardize',
                zero_mean=True,
                unit_variance=True)

            # convert to correct data format
            if K.image_data_format() == 'channels_last':
                axes = 'b01c'
            else:
                axes = 'bc01'
            depth = img_utils.dimshuffle(depth, '01c', axes)

            # repeat if sampling is enabled
            if n_samples > 1:
                depth = np.repeat(depth, n_samples, axis=0)

        # prepare rgb input
        if input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
            # load
            rgb = img_utils.load(rgb_filepath)

            # create mask
            # if args.input_type == INPUT_RGB:
            #     # derive mask from rgb image
            #     mask = rgb > 0
            #     mask_resized = pre.resize_mask(mask.astype('uint8')*255,
            #                                    shape) > 0
            # else:
            #     # mask rgb image using mask derived from depth image
            #    rgb = pre.mask_img(rgb, mask)

            # resize
            rgb = pre.resize_depth_img(rgb, shape)

            # preprocess
            rgb = pre.preprocess_img(
                rgb,
                mask=mask_resized,
                scale01=input_preprocessing == 'scale01',
                standardize=input_preprocessing == 'standardize',
                zero_mean=True,
                unit_variance=True)

            # convert to correct data format
            if K.image_data_format() == 'channels_last':
                axes = 'b01c'
            else:
                axes = 'bc01'
            rgb = img_utils.dimshuffle(rgb, '01c', axes)

            # repeat if sampling is enabled
            if n_samples > 1:
                rgb = np.repeat(rgb, n_samples, axis=0)

        # return preprocessed images
        if input_type == INPUT_DEPTH_AND_RGB:
            return depth, rgb
        elif input_type == INPUT_DEPTH:
            return depth,
        else:
            return rgb,

    # define postprocessing function ------------------------------------------
    def postprocess(output):
        if output_type == OUTPUT_BITERNION:
            return post.biternion2deg(output)
        elif output_type == OUTPUT_REGRESSION:
            return post.rad2deg(output)
        else:
            return post.class2deg(np.argmax(output, axis=-1), n_classes)

    # process files -----------------------------------------------------------
    len_cnt = len(str(len(filepaths)))
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    #fig2 = plt.figure()
    for i, inputs in enumerate(filepaths):
        #print("[{:0{}d}/{:0{}d}]: {}".format(i+1, len_cnt, len(filepaths),
        #                                    len_cnt, inputs))
        print(inputs[1: 2])
        # load and preprocess inputs
        nw_inputs = load_and_preprocess(inputs)

        # predict
        nw_output = model.predict(nw_inputs, batch_size=n_samples)

        # postprocess output
        output = postprocess(nw_output)
        #<< --------- Salida ----------------------------------------------------->>

        print("-------- Salida ----> " + str(np.mean(output)))
        
        #time.sleep(5)
        # visualize inputs and predicted angle
        plt.clf()
        # visualize inputs
        for j, inp in enumerate(nw_inputs):
            # first element of input batch
            img = inp[0]
            
            # convert to 01c
            if K.image_data_format() == 'channels_last':
                axes = '01c'
            else:
                axes = 'c01'
            img = img_utils.dimshuffle(img, axes, '01c')

            # inverse preprocessing
            img = pre.preprocess_img_inverse(
                img,
                scale01=input_preprocessing == 'scale01',
                standardize=input_preprocessing == 'standardize',
                zero_mean=True,
                unit_variance=True)
            # show
            ax = fig.add_subplot(1, len(nw_inputs) + 1, j + 1)
            if img.shape[-1] == 1:
                ax.imshow(img[:, :, 0], cmap='gray',
                          vmin=img[img != 0].min(), vmax=img.max())
            else:
                ax.imshow(img)
            ax.axis('off')

        # visualize output
        ax = fig.add_subplot(1, len(nw_inputs)+1, len(nw_inputs)+1, polar=True)
        ax.set_theta_zero_location('S', offset=0)
        ax.hist(np.deg2rad(output), width=np.deg2rad(2), density=True,
                alpha=0.5 if n_samples > 1 else 1.0, color='#b62708')
        if n_samples > 1:
            mean_rad = circmean(np.deg2rad(output))
            std_rad = circstd(np.deg2rad(output))
            x = np.deg2rad(np.linspace(0, 360, 360))
            pdf_values = norm.pdf(x, mean_rad, std_rad)
            ax.plot(x, pdf_values, color='#b62708', zorder=2, linewidth=5)#b62708--1f77b4
            ax.fill(x, pdf_values, color='#b62708', zorder=2, alpha=0.3)
        ax.set_yscale('symlog')
        ax.set_ylim([0, 20])
        #-> se agrego el textbox ---------------------------------------
        buttonax = plt.axes([0.9, 0.025, 0.2, 0.04])
        txtDistania = TextBox(buttonax, "Distancia", "", "white", "grey")
        txtDistania.set_val(str(output[0]))#carga el valor de la salida
        #----------------------------------------------------------------
        plt.text(0.23, 1.2, 'BACKWARDS', color='k', size='medium', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
        plt.text(0.18, -0.25, 'FRONTWARDS ', color='k', size='medium', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
        plt.text(1.2, 0.48, 'LEFT', color='k', size='medium', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
        plt.text(-0.40, 0.48, 'RIGHT', color='k', size='medium', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
        plt.tight_layout()
        #plt.savefig(f'./img{i}.png', bbox_inches='tight', dpi=75)
        plt.pause(0.0005)


if __name__ == '__main__':
    main()
