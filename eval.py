import argparse
import os

import crossView

import numpy as np

import torch
from torch.utils.data import DataLoader

import cv2
import tqdm

from utils import mean_IU, mean_precision
from opt import get_eval_args as get_args

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_model(models, model_path):
    """Load model(s) from disk
    """
    model_path = os.path.expanduser(model_path)

    assert os.path.isdir(model_path), \
        "Cannot find folder {}".format(model_path)
    print("loading model from folder {}".format(model_path))

    for key in models.keys():
        print("Loading {} weights...".format(key))
        path = os.path.join(model_path, "{}.pth".format(key))
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {
            k: v for k,
                     v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)
    return models


def evaluate():
    opt = get_args()

    # Loading Pretarined Model
    models = {}
    models["encoder"] = crossView.Encoder(18, opt.height, opt.width, True)
    models['CycledViewProjection'] = crossView.CycledViewProjection(in_dim=8)
    models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)

    models["decoder"] = crossView.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, opt.num_class)
    models["transform_decoder"] = crossView.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, opt.num_class, "transform_decoder")

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)

    # Loading Validation/Testing Dataset

    # Data Loaders
    dataset_dict = {"3Dobject": crossView.KITTIObject,
                    "odometry": crossView.KITTIOdometry,
                    "argo": crossView.Argoverse,
                    "raw": crossView.KITTIRAW}

    dataset = dataset_dict[opt.split]
    fpath = os.path.join(
        os.path.dirname(__file__),
        "splits",
        opt.split,
        "{}_files.txt")
    test_filenames = readlines(fpath.format("val"))
    test_dataset = dataset(opt, test_filenames, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        1,
        False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    iou, mAP = np.array([0., 0.]), np.array([0., 0.])
    trans_iou, trans_mAP = np.array([0., 0.]), np.array([0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)
        save_topview(
            inputs["filename"],
            outputs["topview"],
            os.path.join(
                opt.out_dir,
                'topview',
                "{}.png".format(inputs["filename"][0])))
        pred = np.squeeze(
            torch.argmax(
                outputs["topview"].detach(),
                1).cpu().numpy())
        trans_pred = np.squeeze(
            torch.argmax(
                outputs["transform_topview"].detach(),
                1).cpu().numpy())
        true = np.squeeze(inputs[opt.type + "_gt"].detach().cpu().numpy())
        iou += mean_IU(pred, true)
        mAP += mean_precision(pred, true)
        trans_iou += mean_IU(trans_pred, true)
        trans_mAP += mean_precision(trans_pred, true)
    iou /= len(test_loader)
    mAP /= len(test_loader)

    trans_iou /= len(test_loader)
    trans_mAP /= len(test_loader)

    print("Evaluation Results: mIOU: %.4f mAP: %.4f" % (iou[1], mAP[1]))


def process_batch(opt, models, inputs):
    outputs = {}
    # print(inputs["filename"])
    for key, input_ in inputs.items():
        if key != "filename":
            inputs[key] = input_.to("cuda")

    features = models["encoder"](inputs["color"])

    # Cross-view Transformation Module
    transform_feature, retransform_features = models["CycledViewProjection"](features)
    features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

    outputs["topview"] = models["decoder"](features)
    outputs["transform_topview"] = models["transform_decoder"](transform_feature)

    return outputs


def save_topview(idx, tv, name_dest_im):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
    true_top_view[tv_np[1] > tv_np[0]] = 255
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, true_top_view)

    # print("Saved prediction to {}".format(name_dest_im))


if __name__ == "__main__":
    evaluate()
