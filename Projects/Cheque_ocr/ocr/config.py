import os
pwd = os.getcwd()
GPU = True ##OCR 是否启用GPU
GPUID=0 ##调用GPU序号
nmsFlag='gpu' ## cython/gpu/python ##容错性 优先启动GPU，其次是cpython 最后是python
if not GPU:
    nmsFlag='cython'
LSTMFLAG = True
ocrModel = os.path.join(pwd,"weights","ocr-lstm.pth")

