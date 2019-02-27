import os
import time
import shutil
import torch
import torch.autograd
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torchvision.transforms as transforms
from model import Model
from dataset import MyDataset

class Solver:
    def __init__(self,args):
        self.args=args

    def train_model(self):
        global best_acc1
        # build model
        model=Model(self.args.num_classes)
        # DataParrallel will divide and allocate batch_size to all available GPUs
        model=torch.nn.DataParallel(model).cuda()

        # define loss and optimizer
        loss_calc=nn.CrossEntropyLoss().cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=self.args.lr,
                                   weight_decay=self.args.weight_decay)
        # optionally resume from a checkpoint
        if self.args.resume_file:
            if os.path.isfile(self.args.resume_file):
                print("=> loading checkpoint '{}'".format(self.args.resume_file))
                checkpoint=torch.load(self.args.resume_file)
                self.args.start_spoch=checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.args.resume_file,self.args.start_spoch))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume_file))

        # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware
        torch.backends.cudnn.benchmark = True

        # load data
        train_img_dir=os.path.join(self.args.data_path,'train/images')
        val_img_dir=os.path.join(self.args.data_path,'val/images')
        train_txt_file=os.path.join(self.args.data_path,'caffe/train_100classes.txt')
        val_txt_file=os.path.join(self.args.data_path,'caffe/val_100classes.txt')

        transform=transforms.Compose([
            transforms.RandomResizedCrop(self.args.img_size,scale=(0.5,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

        train_dataset=MyDataset(img_txt_file=train_txt_file,img_dir=train_img_dir,transform=transform)
        val_dataset=MyDataset(img_txt_file=val_txt_file,img_dir=val_img_dir,transform=transform)

        train_loader=torch.utils.data.DataLoader(
            dataset=train_dataset,batch_size=self.args.batch_size,shuffle=True,
            num_workers=self.args.load_workers,pin_memory=True)
        val_loader=torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.load_workers, pin_memory=True)


        # train the model
        for epoch in range(self.args.start_epoch,self.args.num_epochs):
            # adjust learning rate
            self.adjust_lr(optimizer,epoch,self.args)

            # train for one epoch
            self.train_epoch(train_loader,model,loss_calc,optimizer,epoch,self.args)

            # evaluate on validation set
            acc1=self.validate(val_loader,model,loss_calc,self.args)

            # remember the best acc1 and save checkpoint
            self.is_best=(acc1>best_acc1)
            best_acc1=max(acc1,best_acc1)
            self.save_checkpoint({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'best_acc1':best_acc1,
                'optimizer':optimizer.state_dict(),
            },self.is_best)

    def train_epoch(self,train_loader,model,loss_calc,optimizer,epoch,args):
        batch_time=AverageMeter()
        data_load_time=AverageMeter()
        losses=AverageMeter()
        top1=AverageMeter()
        top5=AverageMeter()

        # Sets the module in training mode.
        model.train()

        end_time=time.time()

        for i,(input,target) in enumerate(train_loader):
            # measure data loading time
            data_load_time.update(time.time()-end_time)

            input=input.cuda(non_blocking=True)
            target=target.cuda(non_blocking=True)

            # compute output
            output=model(input)
            loss=loss_calc(output,target)

            # measure accuracy and record loss
            acc1,acc5=self.acc_calc(output,target,topk=(1,5))
            losses.update(loss.item(),input.size(0))
            top1.update(acc1[0],input.size(0))
            top5.update(acc5[0],input.size(0))

            # compute gradient and do Adam step
            optimizer.zero_grad()  # Sets gradients of all model parameters to zero.
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time()-end_time)
            end_time=time.time()

            if i%args.print_freq==0:
                print('epoch: {0} step: [{1}/{2}]\n'
                      'batch time: {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\n'
                      'data loading time: {data_load_time.val:.3f} (avg: {data_load_time.avg:.3f})\n'
                      'loss: {loss.val:.4f} (avg: {loss.avg:.4f})\n'
                      'top1 accuracy: {top1.val:.3f} (avg: {top1.avg:.3f})\n'
                      'top5 accuracy: {top5.val:.3f} (avg: {top5.avg:.3f})'.format(
                        epoch,i,len(train_loader),batch_time=batch_time,
                        data_load_time=data_load_time,loss=losses,top1=top1,top5=top5))

        # write to log file
        try:
            with open('train_results.txt','w') as f:
                f.write('epoch: {0} \n'
                  'batch time: {batch_time.avg:.3f}\n'
                  'data loading time: {data_load_time.avg:.3f}\n'
                  'loss: {loss.avg:.4f}\n'
                  'top1 accuracy: {top1.avg:.3f}\n'
                  'top5 accuracy: {top5.avg:.3f}'.format(
                    epoch,batch_time=batch_time,
                    data_load_time=data_load_time,loss=losses,top1=top1,top5=top5))
        except Exception as e:
            print(e)

    def validate(self,val_loader,model,loss_calc,args):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Sets the module in evaluation mode.
        model.eval()

        with torch.no_grad():
            end_time=time.time()

            for i,(input,target) in enumerate(val_loader):
                input=input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output=model(input)
                loss=loss_calc(output,target)

                # meassure accuracy and record loss
                acc1,acc5=self.acc_calc(output,target,topk=(1,5))
                losses.update(loss.item(),input.size(0))
                top1.update(acc1[0],input.size(0))
                top5.update(acc5[0],input.size(0))

                # measure elapsed time
                batch_time.update(time.time()-end_time)
                end_time=time.time()

                if i%args.print_freq==0:
                    print('val step: [{0}/{1}]\n'
                          'batch time: {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\n'
                          'loss: {loss.val:.4f} (avg: {loss.avg:.4f})\n'
                          'top1 accuracy: {top1.val:.3f} (avg: {top1.avg:.3f})\n'
                          'top5 accuracy: {top5.val:.3f} (avg: {top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5))

            # write to log file
            try:
                with open('val_results.txt','w') as f:
                    f.write('loss: {loss.avg:.4f})\n'
                          'top1 accuracy: {top1.avg:.3f}\n'
                          'top5 accuracy: {top5.avg:.3f}'.format(
                        loss=losses, top1=top1, top5=top5))
            except Exception as e:
                print(e)
        return top1.avg


    def adjust_lr(self, optimizer, epoch, args):
        """Set the learning rate to the initial LR decayed by 0.5 every 10 epochs"""
        lr = args.lr * (0.5 * (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def acc_calc(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk=max(topk)
            batch_size=target.size(0)

            _,pred=output.topk(maxk,1,True,True) # here maxk=5
            pred=pred.t()  # transpose!!
            correct=pred.eq(target.view(1,-1).expand_as(pred))
            # eg: num_classes=10 , batch_size=2
            # pred=[[5,1,7,0,2],     target=[[1],   target.view(1,-1).expand_as(pred)=[[1,1,1,1,1],
            #       [2,6,4,7,9]]             [3]]                                      [3,3,3,3,3]]
            # correct=[[0,1,0,0,0],
            #          [0,0,0,0,0]]    #Note the example above has not been transposed yet.
            acc=[]
            for k in topk:
                correct_k=correct[:k].view(-1).float().sum(0,keepdim=True)
                acc.append(correct_k.mul_(100.0/batch_size))
        return acc

    def save_checkpoint(self,state,is_best,filename='checkpoint.pth.tar'):
        torch.save(state,filename)
        if self.is_best:
            shutil.copyfile(filename,'model_best.pth.tar')

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count