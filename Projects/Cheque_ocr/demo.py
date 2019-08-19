import glob
import os
import time
import torch
import json

from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
import cv2
import re
from ocr.crnn.crnn_torch import crnnOcr as crnnOcr



@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir):
    class_names = VOCDataset.class_names
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.bmp'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
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
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        text= ['__background__']
        resDic = {}
        for j in range(len(boxes)):
            xmin = int(boxes[j, 0])
            ymin = int(boxes[j, 1])
            xmax = int(boxes[j, 2])
            ymax = int(boxes[j, 3])

            if labels[j] == 1:
                xmin += 140
                xmax -=130
            elif labels[j] == 2:
                xmin += 130
            elif labels[j] == 4:
                xmin += 40

            hight = ymax - ymin
            width = xmax - xmin

            cropImg = image[ymin: ymin + hight, xmin: xmin + width]
            cropImg = local_threshold(cropImg)

            boxes[j, 0] = xmin
            boxes[j, 1] = ymin
            boxes[j, 2] = xmax
            boxes[j, 3] = ymax
            text_tmp = crnnOcr(Image.fromarray(cropImg))

            if labels[j] == 2:
                text_tmp = re.sub('[^\x00-\xff]', '/', text_tmp)

            text.append(text_tmp)
            resDic[class_names[labels[j]]] = text_tmp

        result = json.dumps(resDic, ensure_ascii=False)
        print(result)

        # drawn_image = draw_boxes(image, boxes, labels, None, text).astype(np.uint8)
        # Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


# 局部阈值
def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8)
    return binary


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.9)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)
    print(args.opts)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print("Loaded configuration file {}".format(args.config_file))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir)


if __name__ == '__main__':
    main()