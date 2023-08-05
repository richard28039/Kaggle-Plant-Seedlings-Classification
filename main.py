import argparse
from utils import *

from torchvision import models

from dataset import *
from train import train
from test import test
from models.resnet import CustomizedResNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='config.yaml', help='hyper parmeter')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--train_path', default='runs/result', help='save to project/name')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()


    utils_function.freeze_key(utils_function.r_yaml(opt.hyp)['seed'])

    
    layers_to_freeze = []
    pretrained_resnet50 = models.resnet50(pretrained=True)
    model = CustomizedResNet(pretrained_resnet50, num_classes=12, freeze_layers=layers_to_freeze)
 
    path_dir = utils_function.save_resul(opt.train_path, opt.name, opt.exist_ok, opt.save_txt) 

    train(opt, model, utils_function.r_yaml(opt.hyp), path_dir)

    test(model, utils_function.r_yaml(opt.hyp), path_dir)
