import os
import argparse
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as TF

from model import SRCNN
from utils import *

parser = argparse.ArgumentParser(description="SRCNN super res toolkit")
parser.add_argument("filename", metavar="LR_image", type=str, help="the path to input LR image")
parser.add_argument("--checkpoint", metavar="path_to_checkpoint", type=str, required=True,
                    help="the path to checkpoint")
parser.add_argument("--cuda", action="store_true", help="use CUDA to speed up computation")
opt = parser.parse_args()

# check cuda
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
# load LR image
open_image = lambda x: TF.to_tensor(Image.open(x).convert("RGB"))
image = open_image(opt.filename).unsqueeze(0)

# load model
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
map_location = "cuda:0" if opt.cuda else device
checkpoint = load_checkpoint(opt.checkpoint, map_location)
# init model
model = SRCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
# process input
with torch.no_grad():
    image = image.to(device)
    output = model(image)
# save result
output_name = os.path.splitext(opt.filename)
output_name = "_srcnn_x3".join(output_name)
torchvision.utils.save_image(output[0], output_name)

print("Output HR image saved to '{output}'".format(output=output_name))
