import web
import base64
import uuid
import numpy as np
import cv2
from predict import run_demo, creat_model
from ssd.config import cfg

url=('/simpleocr','SimpleOCR')

cfg_dir = 'configs/ssd300_voc0712.yaml'
cfg.merge_from_file(cfg_dir)
cfg.merge_from_list([])
cfg.freeze()
print("Loaded configuration file {}".format(cfg_dir))

ckpt = 'weights/model_final.pth'
model = creat_model(cfg, ckpt)

score_threshold = 0.9
output_dir = 'demo/result'


class SimpleOCR:
    def POST(self):
        info = web.input()
        data = info.get('img')#.encode('ascii')
        length = len(data)
        data = data.replace("%3D", "=", length)
        data = data.replace("%2F", "/", length)

        data = data.replace("%2B", "+", length)
        data = data.replace("%0A", "\n", length)
        data = data.replace("%0D", "", length)
        data = data.split('data=')[-1]
        data = bytes(data, encoding="utf8")

        img = base64.b64decode(data)  # base64解码)
        jobid = uuid.uuid1().__str__()
        path = '/tmp/{}.bmp'.format(jobid).format(jobid)
        with open(path,'wb') as f:
            f.write(img)

        return run_demo(cfg, model, score_threshold, path, output_dir)


if __name__ == "__main__":
    app = web.application(url, globals())
    app.run()
