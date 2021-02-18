SqueezeNet PyTorch Implementation for CIFAR-10 Dataset  
================================================================

A simple pytorch implementation of SqueezeNet from [Iandola *et al.*](https://arxiv.org/abs/1602.07360) for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 

#### Main Dependencies
* pytorch 1.7
* torchvision 0.8.1
* numpy 1.19.2
* scikit-learn 0.23.2

#### Why do I need another SqueezeNet for PyTorch?

You don't. I just had fun doing it :)

#### Fun Facts

* SqueezeNet was designed with ImageNet dataset in mind so the paper assumes that the 
input shape is [224,224,3]. CIFAR-10 input shape is [32,32,3] so you can choose which one do you like more. For the latter I kept the original structure intact though changing only `avgpool10` spatial dim

* SqueezeNet design didn't opted out from using fully conected layer and empasized that `conv10 + avgpool10` should be enough. You use the same paradigm by changing the original conv10 from 
`torch.nn.Conv2d(512, 1000, 1)` to `Conv2d(512, num_classes, 1)` or to use an additional fc-layer `self.fc = torch.nn.Linear(in_features=1000, out_features=num_classes)` after the original `conv10`

* As per "design explorations" section of the paper, 3 variation of SqueezeNet are supported, namely 
    * vanilla
    * simple bypass and 
    * complex bypass

