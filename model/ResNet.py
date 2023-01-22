import torch.nn as nn
import math
from.BasicModule import BasicModule
import clip
import torch


model_urls = {
    'resnet18': "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    #'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class CLIPModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.text_shape = 512

    def get_clip_features(self,text):
        #image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            #image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
        return text_features #image_features,

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, clip_addition=None): #clip_addition is the size of the CLIP features, the third dimension of the tensor
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.clip_flag = False
        if clip_addition is not None:
            self.fc_clip_addition = nn.Linear(clip_addition, planes) #way to add CLIP features to the model
            self.clip_flag = True
            #TODO check the size of the CLIP features

    def forward(self, x):
        if self.clip_flag:
            clip_features = x[1]
            residual = x[0]
            input = x[0]
        else:
            clip_features = None
            residual = x
            input = x
        #TODO maybe it coming back with lass than 4 dimensions

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        if clip_features is not None:
            clip_features_fc = self.fc_clip_addition(clip_features.float())
            #pad the clip features to the same size as the residual
            clip_features_fc = clip_features_fc.unsqueeze(2).unsqueeze(3)
            clip_features_fc = clip_features_fc.expand(clip_features_fc.size(0), clip_features_fc.size(1), residual.size(2), residual.size(3)) #TODO check if this is the right way to do it
            out = out + residual + clip_features_fc#self.fc_clip_addition(clip_features) #TODO change clip features to be with gradient?
        else:
            out = out + residual
        out = self.relu(out)

        if clip_features is not None:
            return out,clip_features
        else:
            return out

#
# class Bottleneck(nn.Module): #TODO add to the other models
#     expansion = 4
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual #add here CLIP additon
#         out = self.relu(out)
#
#         return out

config = {
    'A': [2, 2, 2, 2],
    'B': [3, 4, 6, 3],
    'C': [3, 4, 23, 3],
    'D': [3, 8, 36, 3],
}

class ResNet(BasicModule):
    def __init__(self, depth, num_classes=1000, clip_flag=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.model_name = 'resnet'
        assert (depth == 18 or depth == 34 or depth == 50 or depth == 101 or depth == 152)
        if depth == 18:
            block = BasicBlock
            layers = config['A']
        elif depth == 34:
            block = BasicBlock
            layers = config['B']
        # elif depth == 50:
        #     block = Bottleneck
        #     layers = config['B']
        # elif depth == 101:
        #     block = Bottleneck
        #     layers = config['C']
        # else:
        #     block = Bottleneck
        #     layers = config['D']

        self.clip_model = None
        if clip_flag:
            self.clip_model = CLIPModel(model_name='ViT-B/32')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # if clip_flag is not None:
        #     self.fc_clip_addition = nn.Linear(512, planes) #way to add CLIP features to the model
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if self.clip_model is not None: #addition to add CLIP features
            layers.append(block(self.inplanes, planes, stride, downsample, self.clip_model.text_shape)) #TODO change the clip addition
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if self.clip_model is not None:  # addition to add CLIP features
                layers.append(block(self.inplanes, planes, clip_addition=self.clip_model.text_shape)) #TODO do i need it?
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_all): #get the original input
        #call CLIP features
        if self.clip_model is not None:
            x = x_all[0]
            text_x = x_all[1]
            self.clip_addition = self.clip_model.get_clip_features(text_x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)



            x = self.layer1((x,self.clip_addition))
            x = self.layer2((x[0],self.clip_addition))
            x = self.layer3((x[0],self.clip_addition))
            x = self.layer4((x[0],self.clip_addition))
            x = x[0]
        else:
            x = x_all
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
