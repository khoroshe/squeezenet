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

* SqueezeNet design opted out from using fully conected layer and empasized that `conv10 + avgpool10` should be enough. You can use the same paradigm by changing the original conv10 from 
`torch.nn.Conv2d(512, 1000, 1)` to `Conv2d(512, num_classes, 1)` or to use an additional fc-layer `self.fc = torch.nn.Linear(in_features=1000, out_features=num_classes)` after the original `conv10`

* As per "design explorations" section of the paper, 2 first variations of SqueezeNet are supported<sup>1</sup> namely: 
    * vanilla squeezenet, and
    * squeezenet with simple bypass

<sup>1</sup> squeezenet with complex bypass did not yield better results in the original paper so I decided not to implement it

#### Results

A number of design exploration around the optimal learning rate, fc layer usage and preferred image size was carried out.
It was found that using larger image sizes (224x224x3) and ImageNet-like preprocessing was beneficial in terms of the validation accuracy as shown below. The inclusion of the fully connected Linear layer provided marginal improvement in "vanilla, 224x224x3" case.

| Design & Preprocessing | Validation Loss | Validation Accuracy
| --- | --- | --- |
| vanilla, 224x224x3, w/o fc | 0.611 | 0.811 |
| vanilla, 32x32x3, w/o fc | 1.112 | 0.608 |
| vanilla, 224x224x3, with fc | 0.591 | 0.814 |
| vanilla, 32x32x3 with fc | 1.112 | 0.603 |

Adding simple bypass architectural change does not improve classifier accuracy:

| Design & Preprocessing | Validation Loss | Validation Accuracy
| --- | --- | --- |
| vanilla, 224x224x3, with fc | 0.591 | 0.814 |
| simple bypass, 224x224x3, with fc | 0.614 | 0.812 |

F1 Scores:

| Class | F1-Score |
| --- | --- |
|plane|0.815|
|car|0.895|
|bird|0.725|
|cat|0.667|
|deer|0.783|
|dog|0.760|
|frog|0.849|
|horse|0.841|
|ship|0.891|
|truck|0.886| 
