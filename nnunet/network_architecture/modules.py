
# # UpCat(nn.Module) for U-net UP convolution
import torch
from torch import nn


class UpCat(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpCat, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # print('\\\\')
        # print(inputs.size())
        # print(down_outputs.size())
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        # print(inputs.size())
        # print(outputs.size())
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            # addition = torch.rand((outputs.size()[0], outputs.size()[1], offset), out=None).cuda()
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            # print('+++')
            # print(addition.size())
            outputs = torch.cat([outputs, addition], dim=3)
            addition = torch.rand((outputs.size()[0], outputs.size()[1], offset, outputs.size()[3]), out=None).cuda()
            # print('+++')
            # print(addition.size())
            outputs = torch.cat([outputs, addition], dim=2)

        offset1 = inputs.size()[3] - outputs.size()[3]
        # if offset1 < 0:
        #     outputs.resize(outputs, [inputs.size()[0], inputs.size()[1], inputs.size()[2], inputs.size()[3]])
        if offset1 < 0:
            addition = torch.rand((inputs.size()[0], inputs.size()[1], inputs.size()[2], -offset), out=None).cuda()
            # print('+++')
            # print(addition.size())
            inputs = torch.cat([inputs, addition], dim=3)
            addition = torch.rand((inputs.size()[0], inputs.size()[1], -offset, inputs.size()[3]), out=None).cuda()
            # print('+++')
            # print(addition.size())
            inputs = torch.cat([inputs, addition], dim=2)

        # print(inputs.size())
        # print(outputs.size())
        out = torch.cat([inputs, outputs], dim=1)

        return out


class UpCat(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpCat, self).__init__()

        self.infeat = in_feat
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # print('\\\\')
        # print(inputs.size())
        # print(down_outputs.size())
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        # print(inputs.size())
        # print(outputs.size())
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            # addition = torch.rand((outputs.size()[0], outputs.size()[1], offset), out=None).cuda()
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            # print('+++')
            # print(addition.size())
            outputs = torch.cat([outputs, addition], dim=3)
            addition = torch.rand((outputs.size()[0], outputs.size()[1], offset, outputs.size()[3]), out=None).cuda()
            # print('+++')
            # print(addition.size())
            outputs = torch.cat([outputs, addition], dim=2)

        offset1 = inputs.size()[3] - outputs.size()[3]
        # if offset1 < 0:
        #     outputs.resize(outputs, [inputs.size()[0], inputs.size()[1], inputs.size()[2], inputs.size()[3]])
        if offset1 < 0:
            addition = torch.rand((inputs.size()[0], inputs.size()[1], inputs.size()[2], -offset), out=None).cuda()
            # print('+++')
            # print(addition.size())
            inputs = torch.cat([inputs, addition], dim=3)
            addition = torch.rand((inputs.size()[0], inputs.size()[1], -offset, inputs.size()[3]), out=None).cuda()
            # print('+++')
            # print(addition.size())
            inputs = torch.cat([inputs, addition], dim=2)

        # print(inputs.size())
        # print(outputs.size())
        out = torch.cat([inputs, outputs], dim=1)

        return out


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class UpAdd(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpAdd, self).__init__()
        self.infeat = in_feat
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(out_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # print('\\\\')
        # print(inputs.size())
        # print(down_outputs.size())
        outputs = self.up(down_outputs)
        if self.infeat == 96:
            outputs = self.up1(outputs)

        offset = inputs.size()[3] - outputs.size()[3]
        # print(inputs.size())
        # print(outputs.size())
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            # addition = torch.rand((outputs.size()[0], outputs.size()[1], offset), out=None).cuda()
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            # print('+++')
            # print(addition.size())
            outputs = torch.cat([outputs, addition], dim=3)
            addition = torch.rand((outputs.size()[0], outputs.size()[1], offset, outputs.size()[3]), out=None).cuda()
            # print('+++')
            # print(addition.size())
            outputs = torch.cat([outputs, addition], dim=2)

        offset1 = inputs.size()[3] - outputs.size()[3]
        # if offset1 < 0:
        #     outputs.resize(outputs, [inputs.size()[0], inputs.size()[1], inputs.size()[2], inputs.size()[3]])
        if offset1 < 0:
            addition = torch.rand((inputs.size()[0], inputs.size()[1], inputs.size()[2], -offset), out=None).cuda()
            # print('+++')
            # print(addition.size())
            inputs = torch.cat([inputs, addition], dim=3)
            addition = torch.rand((inputs.size()[0], inputs.size()[1], -offset, inputs.size()[3]), out=None).cuda()
            # print('+++')
            # print(addition.size())
            inputs = torch.cat([inputs, addition], dim=2)

        # print(inputs.size())
        # print(outputs.size())
        out = inputs + outputs

        return out
