import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, build_model
from detectron2.modeling.backbone import FPN
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
setup_logger()

import numpy as np
import cv2
import random
import os
import json
import argparse
import glob
import multiprocessing as mp
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from predictor import VisualizationDemo

class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

def label():
    with open("train.json") as f:
        imgs_anns = json.load(f)
    lb = []
    for cat in imgs_anns['categories']:
        lb.append(cat['name'])
    return lb

def get_traffic_signs(json_file):
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    annos = imgs_anns['annotations']
    for idx, f in enumerate(imgs_anns['images']):
        record = {}
        filename = os.path.join("JPEGImages", f['file_name'])
        height = f['height']
        width = f['width']
        record["file_name"] = filename
        record["image_id"] = f['id']
        record["height"] = height
        record["width"] = width
        objs = []
        for anno in annos:
            # if anno['image_id'] > f['id']:
            #     break
            if anno['image_id'] == f['id']:
                poly = anno['segmentation']
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": poly,
                    "category_id": anno['category_id']
                }
                objs.append(obj)
        if len(objs) == 0:
            continue
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def evaluation(cfg, model, model_name):
    evaluator = COCOEvaluator("traffic_test", cfg, False, output_dir="./output-%s/" % model_name)
    val_loader = build_detection_test_loader(cfg, "traffic_test")
    inference_on_dataset(model, val_loader, evaluator)

def infer_img(cfg, predictor, fname=None):
    if fname:
        testim = cv2.imread(fname)
        outputs = predictor(testim)
        v = Visualizer(testim[:, :, ::-1],
                        metadata=traffic_metadata, 
                        scale=0.8, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("test", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    else:
        dataset_dicts = get_traffic_signs("test.json")
        for d in random.sample(dataset_dicts, 3):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                        metadata=traffic_metadata, 
                        scale=0.8, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # for i in range(len(outputs["instances"]["pred_classes"])):
            #     print("%s: %f" % (outputs["instances"]["pred_classes"][i], outputs["instances"]["pred_classes"]["scores"]))
            cv2.imshow("test", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

def infer_video(cfg, video_name, output=None):
    video = cv2.VideoCapture(video_name)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(height)
    basename = os.path.basename(video_name)

    if output:
        try:
            os.remove(output)
        except:
            pass
        if os.path.isdir(output):
            output_fname = os.path.join(output, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".mkv"
        else:
            output_fname = output
        assert not os.path.isfile(output_fname), output_fname
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
    # assert os.path.isfile(args.video_input)
    demo = VisualizationDemo(cfg)
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        if output:
            output_file.write(vis_frame)
        else:
            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            cv2.imshow(basename, vis_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    video.release()
    if output:
        output_file.release()
    else:
        cv2.destroyAllWindows()

def test_dataset(traffic_metadata):
    dataset_dicts = get_traffic_signs("train.json")
    # for d in dataset_dicts[:3]:#random.sample(dataset_dicts, 3):
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=traffic_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("test", out.get_image()[:, :, ::-1])
        # cv2.imshow("test", img)
        cv2.waitKey(0)

@BACKBONE_REGISTRY.register()
class StnResNet(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 200)

    def forward(self, x):
        f = {}
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f['res1'] = self.resnet.layer1(x)
        x = self.resnet.layer1(x)
        f['res2'] = self.resnet.layer2(x)
        x = self.resnet.layer2(x)
        f['res3'] = self.resnet.layer3(x)
        x = self.resnet.layer3(x)
        f['res4'] = self.resnet.layer4(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return f
    
    def output_shape(self):
        return {
            "res1": ShapeSpec(channels=64, stride=1),
            "res2": ShapeSpec(channels=128, stride=2),
            "res3": ShapeSpec(channels=256, stride=4),
            "res4": ShapeSpec(channels=512, stride=8),
            }

@BACKBONE_REGISTRY.register()
def build_StnResNet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = StnResNet(cfg, input_shape)
    in_features = ["res1", "res2", "res3", "res4"]
    out_channels = 4
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        top_block=LastLevelMaxPool(),
        out_channels=out_channels,
    )
    return backbone

if __name__ == "__main__":
    for d in ["train", "test"]:
        DatasetCatalog.register("traffic_" + d, lambda d=d: get_traffic_signs(d+".json"))
        MetadataCatalog.get("traffic_" + d).set(thing_classes=label())
    traffic_metadata = MetadataCatalog.get("traffic_train")
    # test_dataset(traffic_metadata)

    for model in ["mask_rcnn_R_50_C4_3x", "mask_rcnn_R_50_DC5_3x", "mask_rcnn_R_50_FPN_3x",
                  "mask_rcnn_R_101_C4_3x", "mask_rcnn_R_101_DC5_3x", "mask_rcnn_R_101_FPN_3x"]:
        cfg = get_cfg()
        if model == "stnresnet":
            cfg.MODEL.FPN.IN_FEATURES = ["res1", "res2", "res3", "res4"]
            cfg.MODEL.FPN.OUT_CHANNELS = 4
            cfg.MODEL.BACKBONE.NAME = "build_StnResNet_fpn_backbone"
        else:
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/%s.yaml" % model))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/%s.yaml" % model)  # Let training initialize from model zoo
        cfg.DATASETS.TRAIN = ("traffic_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 20000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 200
        cfg.OUTPUT_DIR = "./output-%s/" % model
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=True)
        trainer.train()

        cfg.DATASETS.TEST = ("traffic_test", )
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set the testing threshold for this model
        predictor = DefaultPredictor(cfg)

        # infer_video(cfg, "test1.mp4", "testout001.mkv")
        # infer_video(cfg, "test1.mp4")
        evaluation(cfg, trainer.model, model)
        # infer_img(cfg, predictor, "test2.jpg")
        # infer_img(cfg, predictor)
