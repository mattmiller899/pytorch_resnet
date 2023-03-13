from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import resnet_utils as utils
from functools import partial


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=padding)

class ReshapeTensor(nn.Module):
    def __init__(self, read_size, latent_size, model_type="gen"):
        super().__init__()
        self.read_size = read_size
        self.latent_size = latent_size
        self.model_type = model_type


    def forward(self, x):
        if self.model_type == "gen":
            return x.view(-1, self.latent_size, self.read_size)
        else:
            return x.view(-1, self.latent_size * self.read_size)


class Conv1dAutoPad(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.padding = self.kernel_size[0] // 2


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm1d(out_channels))


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, debug=False):
        super(ResBasicBlock, self).__init__()
        self.debug = debug
        self.conv1 = nn.Sequential(
                        conv3x3(in_channels, out_channels, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        conv3x3(out_channels, out_channels, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        if self.debug:
            print(f"resblock x.size = {x.size()}")
        out = self.conv1(x)
        if self.debug:
            print(f"resblock conv1 out.size = {out.size()}")
        out = self.conv2(out)
        if self.debug:
            print(f"resblock conv2 out.size = {out.size()}")
        if self.downsample:
            residual = self.downsample(x)
        if self.debug:
            print(f"resblock residual.size = {residual.size()}")
        out += residual
        out = self.relu(out)
        return out


class ResBottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, debug=False, base_width=64):
        super(ResBottleneckBlock, self).__init__()
        self.debug = debug
        self.conv1 = nn.Sequential(
                        conv1x1(in_channels, out_channels),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        conv3x3(out_channels, out_channels, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        conv1x1(out_channels, out_channels * self.expansion),
                        nn.BatchNorm2d(out_channels * self.expansion))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        if self.debug:
            print(f"resblock x.size = {x.size()}")
        out = self.conv1(x)
        if self.debug:
            print(f"resblock conv1 out.size = {out.size()}")
        out = self.conv2(out)
        if self.debug:
            print(f"resblock conv2 out.size = {out.size()}")
        out = self.conv3(out)
        if self.debug:
            print(f"resblock conv3 out.size = {out.size()}")
        if self.downsample:
            residual = self.downsample(x)
        if self.debug:
            print(f"resblock residual.size = {residual.size()}")
        out += residual
        out = self.relu(out)
        return out

class ResNetModel(nn.Module):
    """
    The overall ResNet model
    """
    def __init__(self, layers, block_type, vecs, num_kmers, in_channels, use_gpu=True, freeze=True, debug=False, activation="relu", kmer_sizes=None, vec_sizes=None, out_channels=64,
                 *args, **kwargs):
        super(ResNetModel, self).__init__()
        self.DEVICE = torch.device("cuda" if use_gpu else "cpu")
        self.IN_CHANNELS = in_channels
        self.OUT_CHANNELS = out_channels
        self.debug = debug
        self.NUM_KMERS = num_kmers
        self.LAYERS = layers
        self.VEC_SIZES = vec_sizes
        self.MAX_SIZE = max(vec_sizes)
        self.KMER_SIZES = kmer_sizes
        self.VOCAB_SIZE = 0
        for k in kmer_sizes:
            self.VOCAB_SIZE += 4 ** k
        if block_type == "basic":
            block = ResBasicBlock
        elif block_type == "bottleneck":
            block = ResBottleneckBlock
        else:
            print(f"ERROR: invalid type of block {block_type}")
            exit()
        self.embeddings = nn.ModuleList()
        if in_channels == 1:
            self.embeddings.append(nn.Embedding.from_pretrained(vecs, freeze=freeze))
        else:
            for i in range(self.IN_CHANNELS):
                v = vecs[i]
                if self.debug:
                    print(f"v.size = {v.size()}")
                # print(f"{i} v size= {v.size()}\n{i} v = {v}")
                padded = torch.zeros(self.VOCAB_SIZE, self.MAX_SIZE)
                padded[:, :self.VEC_SIZES[i]] = v
                self.embeddings.append(nn.Embedding.from_pretrained(padded, freeze=freeze))

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, self.OUT_CHANNELS, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.OUT_CHANNELS),
            nn.ReLU()
        )
        # Make separate variables cuz OUT_CHANNELS gonna be changing as we run
        self.l0_out = self.OUT_CHANNELS
        self.l1_out = self.OUT_CHANNELS*2
        self.l2_out = self.OUT_CHANNELS*4
        self.l3_out = self.OUT_CHANNELS*8


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l0 = self.ResNetLayer(block, 64, self.LAYERS[0], stride=1)
        print(self.l0)
        self.l1 = self.ResNetLayer(block, 128, self.LAYERS[1], stride=2)
        print(self.l1)
        self.l2 = self.ResNetLayer(block, 256, self.LAYERS[2], stride=2)
        print(self.l2)
        self.l3 = self.ResNetLayer(block, 512, self.LAYERS[3], stride=2)
        print(self.l3)
        exit()
        self.avgpool = nn.AvgPool2d(3, stride=1)
        # TODO figure out how to get this value from input dimensions/model architecture
        self.fc = nn.Linear(17408 * block.expansion, 1)


    def ResNetLayer(self, block, planes, blocks, stride=1):
        """
        A ResNet layer composed by `blocks` blocks stacked one after the other
        """
        downsample = None
        if stride != 1 or self.OUT_CHANNELS != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.OUT_CHANNELS, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.OUT_CHANNELS, planes, stride, downsample, debug=self.debug))
        self.OUT_CHANNELS = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.OUT_CHANNELS, planes, debug=self.debug))
        return nn.Sequential(*layers)


    def forward(self, kmers):
        (batch_size, _) = kmers.size()  # (batch, num_kmers)
        # kmers = kmers.transpose(0, 1) # (batch, num_kmers)
        # if self.debug:
        # print(f"kmers.size = {kmers.size()}")
        if self.IN_CHANNELS > 1:
            emb_chans = [self.embeddings[i](kmers) for i in range(self.IN_CHANNELS)]
            # print(f"emb chans shape = ({len(emb_chans)}, {len(emb_chans[0])}, {len(emb_chans[0][0])}, {len(emb_chans[0][0][0])})")
            emb_chans = torch.stack(emb_chans, 1).to(self.DEVICE)
            # print(f"emb chans size = {emb_chans.size()}")
        else:
            emb_chans = self.embeddings[0](kmers).unsqueeze(1)  # (batch, 1, num_kmers, embedding_size
        # if self.debug:
        #    print(f"\nemb chans size = {emb_chans.size()}\nemb = {emb_chans[:,:,:,0]}")
        x = emb_chans
        if self.debug:
            print(f"init x.size = {x.size()}")
        x = self.conv1(x)
        if self.debug:
            print(f"conv1 x.size = {x.size()}")
        x = self.maxpool(x)
        if self.debug:
            print(f"maxpool x.size = {x.size()}")
        x = self.l0(x)
        if self.debug:
            print(f"l0 x.size = {x.size()}")
        x = self.l1(x)
        if self.debug:
            print(f"l1 x.size = {x.size()}")
        x = self.l2(x)
        if self.debug:
            print(f"l2 x.size = {x.size()}")
        x = self.l3(x)
        if self.debug:
            print(f"l3 x.size = {x.size()}")
        x = self.avgpool(x)
        if self.debug:
            print(f"avgpool x.size = {x.size()}")
        x = x.view(x.size(0), -1)
        if self.debug:
            print(f"flat x.size = {x.size()}")
        x = self.fc(x)
        if self.debug:
            print(f"fc x.size = {x.size()}\nfc x = {x}")
        output = torch.sigmoid(x)
        return output
