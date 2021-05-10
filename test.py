import argparse
import glob
import os

import PIL.Image as pil

import cv2

from crossView import model, CrossViewTransformer, CycledViewProjection

import numpy as np

import torch

from torchvision import transforms

from easydict import EasyDict as edict
import matplotlib.pyplot as PLT


def get_args():
    parser = argparse.ArgumentParser(
        description="Testing options")
    parser.add_argument("--image_path", type=str,
                        help="path to folder of images", required=True)
    parser.add_argument("--model_path", type=str,
                        help="path to MonoLayout model", required=True)
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="extension of images in the folder")
    parser.add_argument("--out_dir", type=str,
                        default="output directory to save topviews")
    parser.add_argument("--type", type=str,
                        default="static/dynamic/both")
    parser.add_argument("--view", type=str, default=1, help="view number")
    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "argo",
            "3Dobject",
            "odometry",
            "raw"],
        help="Data split for training/validation")

    configs = edict(vars(parser.parse_args()))
    return configs


def save_topview(idx, tv, name_dest_im):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
    true_top_view[tv_np[1] > tv_np[0]] = 255
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, true_top_view)

    print("Saved prediction to {}".format(name_dest_im))


def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {
        k: v for k,
        v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    CVP_path = os.path.join(args.model_path, "CycledViewProjection.pth")
    CVP_dict = torch.load(CVP_path, map_location=device)
    models['CycledViewProjection'] = CycledViewProjection(in_dim=8)
    filtered_dict_cvp = {
        k: v for k,
        v in CVP_dict.items() if k in models["CycledViewProjection"].state_dict()}
    models["CycledViewProjection"].load_state_dict(filtered_dict_cvp)

    CVT_path = os.path.join(args.model_path, "CrossViewTransformer.pth")
    CVT_dict = torch.load(CVT_path, map_location=device)
    models['CrossViewTransformer'] = CrossViewTransformer(128)
    filtered_dict_cvt = {
        k: v for k,
        v in CVT_dict.items() if k in models["CrossViewTransformer"].state_dict()}
    models["CrossViewTransformer"].load_state_dict(filtered_dict_cvt)

    decoder_path = os.path.join(args.model_path, "decoder.pth")
    DEC_dict = torch.load(decoder_path, map_location=device)
    models["decoder"] = model.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc)
    filtered_dict_dec = {
        k: v for k,
        v in DEC_dict.items() if k in models["decoder"].state_dict()}
    models["decoder"].load_state_dict(filtered_dict_dec)

    transform_decoder_path = os.path.join(args.model_path, "transform_decoder.pth")
    TRDEC_dict = torch.load(transform_decoder_path, map_location=device)
    models["transform_decoder"] = model.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc)
    filtered_dict_trdec = {
        k: v for k,
        v in TRDEC_dict.items() if k in models["transform_decoder"].state_dict()}
    models["transform_decoder"].load_state_dict(filtered_dict_trdec)

    for key in models.keys():
        models[key].to(device)
        models[key].eval()

    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        if args.split == "argo":
            paths = glob.glob(os.path.join(
            args.image_path, '*/ring_front_center/*.{}'.format(args.ext)))
        else:
            paths = glob.glob(os.path.join(
                args.image_path, '*.{}'.format(args.ext)))

        output_directory = args.out_dir
        try:
            os.mkdir(output_directory)
        except BaseException:
            pass
    else:
        raise Exception(
            "Can not find args.image_path: {}".format(
                args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize(
                (feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            # PREDICTION
            input_image = input_image.to(device)

            features = models["encoder"](input_image)

            transform_feature, retransform_features = models["CycledViewProjection"](features)
            features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

            output_name = os.path.splitext(os.path.basename(image_path))[0]
            print("Processing {:d} of {:d} images- ".format(idx + 1, len(paths)))
            tv = models["decoder"](features, is_training=False)
            transform_tv = models["transform_decoder"](transform_feature, is_training=False)

            save_topview(
                idx,
                tv,
                os.path.join(
                    args.out_dir,
                    args.type,
                    "{}.png".format(output_name)))

    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
