# -*- coding: UTF-8 -*-
import os
import argparse
from solver import Solver

os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4'

parser=argparse.ArgumentParser('ImageNet classification using a Vgg16 network')
parser.add_argument('--img_size',default=224,type=int,help='input image size (default:224)')
parser.add_argument('--num_classes',default=100,type=int,help='number of image classes (default:100)')
parser.add_argument('--batch_size',default=128,type=int,help='batch size (default:64)')
parser.add_argument('--load_workers',default=10,type=int,help='number of subprocesses to use for data loading, \
                                                     "0" means the data will be loaded in the main process (default:10)')
parser.add_argument('--lr',default=1e-4,type=float,help='learning rate (default:1e-4)')
parser.add_argument('--weight_decay',default=1e-5,type=float,help='weight decay of L2 penalty (default:1e-5)')
parser.add_argument('--num_epochs',default=1000,type=int,help='number of training epochs (default:1000)')
parser.add_argument('--data_path',default='/dataset/ImageNet',type=str,help='path to dataset (default:"/dataset/ImageNet")')
parser.add_argument('--resume_file',default='',type=str,help='path to latest checkpoint (default: none)')
parser.add_argument('--print_freq',type=int,default=10,help='print frequency (default: 10)')
parser.add_argument('--start_epoch',type=int,default=0,help='start epoch of training')

def main():
    args=parser.parse_args()

    solver=Solver(args)
    solver.train_model()

if __name__=='__main__':
    main()