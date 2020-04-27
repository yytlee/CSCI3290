#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 4
# Name : Lee Tsz Yan
# Student ID : 1155110177
# Email Addr : 1155110177@link.cuhk.edu.hk
#

################################################################
# you are free to modify this file as long as it keeps functional

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model import SRCNN
from data import *
from utils import *

supress_warning()
# dataloader worker
num_workers = 4
if os.name == "nt":
    print(
        "On windows, you may need to set num_workers=0."
    )
    num_workers = 0

parser = ArgParser(description="Assignment 4: super resolution convolutional neural network")

subparsers = parser.add_subparsers(dest="mode", required=True, help="sub commands")
train_parser = subparsers.add_parser("train", help="train SRCNN model")
resume_parser = subparsers.add_parser("resume", help="resume training")
inspect_parser = subparsers.add_parser("inspect", help="inspect a checkpoint")

train_parser.add_argument("--batch-size", type=int, default=8, help="training batch size")
train_parser.add_argument("--num-epoch", type=int, default=400, help="number of epochs to train for")
train_parser.add_argument("--save-freq", type=int, default=20, help="save checkpoint every SAVE_FREQ epoches")
train_parser.add_argument("--lr", type=float, default=0.0001, help="learning Rate. default=0.001")
train_parser.add_argument("--cuda", action="store_true", help="use cuda to train the model?")
# train_parser.add_argument("--no-progress", action="store_true", help="hide the progress bar?")

resume_parser.add_argument("--batch-size", type=int, help="training batch size")
resume_parser.add_argument("--num-epoch", type=int, help="number of epochs to train for")
resume_parser.add_argument("--save-freq", type=int, help="save checkpoint every SAVE_FREQ epoches")
resume_parser.add_argument("--lr", type=float, help="learning Rate. default=0.001")
resume_parser.add_argument("--cuda", action="store_true", help="use cuda to train the model?")
resume_parser.add_argument("checkpoint", type=str, help="path to checkpoint")
# resume_parser.add_argument("--no-progress", action="store_true", help="hide the progress bar?")

inspect_parser.add_argument("checkpoint", type=str, help="path to checkpoint")

opt = parser.parse_args()

start_epoch = 0
checkpoint = None
if opt.mode == "resume":
    checkpoint = load_checkpoint(opt.checkpoint)
    print("Using checkpoint '{}'".format(opt.checkpoint))
    save_opt = checkpoint["opt"]
    start_epoch = checkpoint["epoch"] + 1
    for arg in vars(opt):
        val = getattr(opt, arg)
        if val == None:
            setattr(opt, arg, getattr(save_opt, arg))
elif opt.mode == "inspect":
    checkpoint = load_checkpoint(opt.checkpoint)
    print("Saved at epoch {} with PSNR={:.4f}".format(checkpoint["epoch"], checkpoint["psnr"]))
    print("Trained with arguments:", checkpoint["opt"])
    exit(0)

# if opt.no_progress:
#     tqdm = tqdm_wrapper

print("Arguments:", opt)
# device
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
map_location = "cuda:0" if opt.cuda else device
# model
model = SRCNN()
if checkpoint:
    checkpoint_ = load_checkpoint(opt.checkpoint, map_location)
    model.load_state_dict(checkpoint_["model_state_dict"])

model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
if checkpoint:
    optimizer.load_state_dict(checkpoint_["optimizer_state_dict"])

# data
train_set = SRDataset(collect_data(data_path["train"], progress=True))
test_set = SRDataset(collect_data(data_path["test"]))

train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True,
                                                num_workers=num_workers)
test_data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False,
                                               num_workers=num_workers)

# loss func
loss_function = nn.MSELoss()

record = {
    "cur": 0,
    "best": 0,
    "epoch": 0,
    "model": None,
    "optim": None
}


# train procedure
def train(epoch, data_loader):
    model.train()
    loss_metric = AvgMetric(len(data_loader))
    for data in tqdm(data_loader, ascii=True):
        input, target = data[0].to(device), data[1].to(device)
        ######################
        # write your code here
        optimizer.zero_grad()
        output = model(input)
        loss = loss_function(output, target)
        loss_metric.add(loss)
        loss.backward()
        optimizer.step()

    print("[Train] Epoch {cur}/{total} complete: Avg. Loss={val:.4f}".format(cur=epoch, total=opt.num_epoch,
                                                                             val=loss_metric.average()))


# test procedure
def test(epoch, data_loader):
    model.eval()
    psnr_metric = AvgMetric(len(data_loader))
    with torch.no_grad():
        for data in data_loader:
            input, target = data[0].to(device), data[1].to(device)
            output = model(input)
            mse = loss_function(output, target)
            psnr = 10 * torch.log10(1 / mse)
            psnr_metric.add(psnr.item())
        record["cur"] = psnr_metric.average()
        if record["cur"] > record["best"]:
            record["best"] = record["cur"]
            record["epoch"] = epoch
            record["model"] = copy.deepcopy(model.state_dict())
            record["optim"] = copy.deepcopy(optimizer.state_dict())
        print("[Test] Epoch {cur}/{total} complete: Avg. PSNR={val:.4f}".format(cur=epoch, total=opt.num_epoch,
                                                                                val=psnr_metric.average()))


# train num_epoch-start_epoch+1 epochs
for epoch in range(start_epoch, opt.num_epoch):
    train(epoch, train_data_loader)
    test(epoch, test_data_loader)
    if (epoch + 1) % opt.save_freq == 0:
        # save a checkpoint
        save_checkpoint(epoch, model, optimizer, opt=opt, psnr=record["cur"])

if record["model"] != None:
    # save best checkpoint
    print("Best PSNR={:.4f}".format(record["best"]))
    save_checkpoint(record["epoch"], record["model"], record["optim"], psnr=record["best"], best=True, opt=opt)
