import torch.nn as nn

class Model(nn.Module):
    def __init__(self,num_classes):
        super(Model,self).__init__()
        # conv block1
        self.layer1=nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            # 1-2 conv layer
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            # 1 pooling layer
            nn.MaxPool2d(kernel_size=2,stride=2))

        # conv block2
        self.layer2=nn.Sequential(
            # 2-1 conv layer
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # 2-1 conv layer
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # 2 pooling layer
            nn.MaxPool2d(kernel_size=2,stride=2))

        # conv block3
        self.layer3=nn.Sequential(
            # 3-1 conv layer
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 3-2 conv layer
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 3-3 conv layer
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 3 pooling layer
            nn.MaxPool2d(kernel_size=2,stride=2))

        # conv block4
        self.layer4=nn.Sequential(
            # 4-1 conv layer
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 4-2 conv layer
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 4-3 conv layer
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 4 pooling layer
            nn.MaxPool2d(kernel_size=2,stride=2))

        # conv block5
        self.layer5=nn.Sequential(
            # 5-1 conv layer
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 5-2 conv layer
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 5-3 conv layer
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 5 pooling layer
            nn.MaxPool2d(kernel_size=2,stride=2))

        # fc6
        self.layer6=nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout())

        # fc7
        self.layer7=nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout())

        # fc8
        self.layer8=nn.Sequential(
            nn.Linear(4096,num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        vgg_features=out.view(out.size(0),-1) # out.size(0) = batch size
        out=self.layer6(vgg_features)
        out=self.layer7(out)
        out=self.layer8(out)

        return out