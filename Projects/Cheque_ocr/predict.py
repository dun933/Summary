
import os
import time
import cv2
import re
import torch
import numpy as np
import json

from PIL import Image
from ssd.data.datasets import VOCDataset
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
from ocr.crnn.crnn_torch import crnnOcr as crnnOcr


def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8)
    return binary


@torch.no_grad()
def creat_model(cfg, ckpt):
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))
    return model


@torch.no_grad()
def run_demo(cfg, model, score_threshold, images_dir, output_dir):
    device = torch.device(cfg.MODEL.DEVICE)
    class_names = VOCDataset.class_names
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    start = time.time()
    image_name = os.path.basename(images_dir)

    image = np.array(Image.open(images_dir).convert("RGB"))
    height, width = image.shape[:2]
    images = transforms(image)[0].unsqueeze(0)
    load_time = time.time() - start

    start = time.time()
    result = model(images.to(device))[0]
    inference_time = time.time() - start

    result = result.resize((width, height)).to(cpu_device).numpy()
    boxes, labels, scores = result['boxes'], result['labels'], result['scores']

    indices = scores > score_threshold
    boxes = boxes[indices]
    labels = labels[indices]
    meters = ' | '.join(
        [
            'objects {:02d}'.format(len(boxes)),
            'load {:03d}ms'.format(round(load_time * 1000)),
            'inference {:03d}ms'.format(round(inference_time * 1000)),
            'FPS {}'.format(round(1.0 / inference_time))
        ]
    )
    print('({:04d}) {}: {}'.format(len(images_dir), image_name, meters))

    text= ['__background__']
    resDic = {}
    for j in range(len(boxes)):
        xmin = int(boxes[j, 0])
        ymin = int(boxes[j, 1])
        xmax = int(boxes[j, 2])
        ymax = int(boxes[j, 3])

        if labels[j] == 1:
            xmin += 140
            xmax -= 130
        elif labels[j] == 2:
            xmin += 130
        elif labels[j] == 4:
            xmin += 40

        hight = ymax - ymin
        width = xmax - xmin

        cropImg = image[ymin: ymin + hight, xmin: xmin + width]
        cropImg = local_threshold(cropImg)

        text_tmp = crnnOcr(Image.fromarray(cropImg))

        if labels[j] == 2:
            text_tmp = re.sub('[^\x00-\xff]', '/', text_tmp)

        text.append(text_tmp)
        resDic[class_names[labels[j]]] = text_tmp

    return json.dumps(resDic, ensure_ascii=False).encode('utf-8')
