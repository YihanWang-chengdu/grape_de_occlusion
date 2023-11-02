# full assembly of the sub-parts to form the complete net
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv,up_without_shortcut,Fusing_without_shortcut

class UNetD2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNetD2, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 32 * w)
        self.up1 = up(64 * w, 16 * w)
        self.up2 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

    def forward(self, x):
        x1 = self.inc(x) # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 32
        x = self.up1(x3, x2) # 16
        x = self.up2(x, x1) # 16
        x = self.outc(x)
        return x


class UNetD3(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNetD3, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 64 * w)
        self.down3 = down(64 * w, 64 * w)
        self.up2 = up(128 * w, 32 * w)
        self.up3 = up(64 * w, 16 * w)
        self.up4 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet_forboundary(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNet_forboundary, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))

        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

        self.up1_forboundary = up(int(192 * w), int(64 * w))
        self.up2_forboundary = up(int(96 * w), int(32 * w))
        self.up3_forboundary = up(int(48 * w), int(16 * w))
        self.up4_forboundary = up(int(32 * w), int(16 * w))
        self.outc_forboundary = outconv(int(16 * w), 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_out1 = self.up1(x5, x4)
        x_out2 = self.up2(x_out1, x3)
        x_out3 = self.up3(x_out2, x2)
        x_out4 = self.up4(x_out3, x1)
        x_out = self.outc(x_out4)

        x_boundary_1 = self.up1_forboundary(x5,x_out1)
        x_boundary_2 = self.up2_forboundary(x_boundary_1, x_out2)
        x_boundary_3 = self.up3_forboundary(x_boundary_2, x_out3)
        x_boundary_4 = self.up4_forboundary(x_boundary_3, x_out4)
        x_boundary_out = self.outc_forboundary(x_boundary_4)
        return x_out,x_boundary_out

# class UNet_forboundary_withoutshortcut(nn.Module):
#     def __init__(self, in_channels=3, w=4, n_classes=2,out_boundary = True):
#         super(UNet_forboundary_withoutshortcut, self).__init__()
#         self.out_boundary = out_boundary
#
#         self.inc = inconv(in_channels, int(16 * w))
#         self.down1 = down(int(16 * w), int(32 * w))
#         self.down2 = down(int(32 * w), int(64 * w))
#         self.down3 = down(int(64 * w), int(128 * w))
#         self.down4 = down(int(128 * w), int(256 * w))
#
#
#         self.up1 = up_without_shortcut(int(256 * w), int(128 * w))
#         self.up2 = up_without_shortcut(int(128 * w), int(64 * w))
#         self.up3 = up_without_shortcut(int(64 * w), int(32 * w))
#         self.up4 = up_without_shortcut(int(32 * w), int(16 * w))
#         self.outc = outconv(int(16 * w), n_classes)
#
#         if self.out_boundary:
#          self.up1_forboundary = up(int(384 * w), int(128 * w))
#          self.up2_forboundary = up(int(192 * w), int(64 * w))
#          self.up3_forboundary = up(int(96 * w), int(32 * w))
#          self.up4_forboundary = up(int(48 * w), int(16 * w))
#          self.outc_forboundary = outconv(int(16 * w), 1)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         x_out1 = self.up1(x5)
#         x_out2 = self.up2(x_out1)
#         x_out3 = self.up3(x_out2)
#         x_out4 = self.up4(x_out3)
#         x_out = self.outc(x_out4)
#
#         if self.out_boundary:
#          x_boundary_1 = self.up1_forboundary(x5,x_out1)
#          x_boundary_2 = self.up2_forboundary(x_boundary_1, x_out2)
#          x_boundary_3 = self.up3_forboundary(x_boundary_2, x_out3)
#          x_boundary_4 = self.up4_forboundary(x_boundary_3, x_out4)
#          x_boundary_out = self.outc_forboundary(x_boundary_4)
#          return x_out,x_boundary_out
#         else:
#          return x_out
class UNet_forboundary_withoutshortcut(nn.Module):
  def __init__(self, in_channels=3, w=4, n_classes=2,out_boundary = True,boundary_with_shared = False):
        super(UNet_forboundary_withoutshortcut, self).__init__()
        self.out_boundary = out_boundary
        self.boundary_with_shared = boundary_with_shared

        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(256 * w))

        if self.out_boundary:
            if self.boundary_with_shared:
                self.up1 = up_without_shortcut(int(256 * w), int(128 * w))  # 1/2
                self.up1_forboundary = up(int(384 * w), int(128 * w))  # 1/2

                self.up2 = Fusing_without_shortcut(int(192 * w), int(64 * w))
                self.up2_forboundary = up(int(192 * w), int(64 * w))

                self.up3 = Fusing_without_shortcut(int(96 * w), int(32 * w))
                self.up3_forboundary = up(int(96 * w), int(32 * w))

                self.up4 = Fusing_without_shortcut(int(48 * w), int(16 * w))
                self.up4_forboundary = up(int(48 * w), int(16 * w))

                self.outc = outconv(int(16 * w), n_classes)
                self.outc_forboundary = outconv(int(16 * w), 1)
            else:
                self.up1 = up_without_shortcut(int(256 * w), int(128 * w))
                self.up2 = up_without_shortcut(int(128 * w), int(64 * w))
                self.up3 = up_without_shortcut(int(64 * w), int(32 * w))
                self.up4 = up_without_shortcut(int(32 * w), int(16 * w))
                self.outc = outconv(int(16 * w), n_classes)

                self.up1_forboundary = up(int(384 * w), int(128 * w))
                self.up2_forboundary = up(int(192 * w), int(64 * w))
                self.up3_forboundary = up(int(96 * w), int(32 * w))
                self.up4_forboundary = up(int(48 * w), int(16 * w))
                self.outc_forboundary = outconv(int(16 * w), 1)
        else:
            self.up1 = up_without_shortcut(int(256 * w), int(128 * w))
            self.up2 = up_without_shortcut(int(128 * w), int(64 * w))
            self.up3 = up_without_shortcut(int(64 * w), int(32 * w))
            self.up4 = up_without_shortcut(int(32 * w), int(16 * w))
            self.outc = outconv(int(16 * w), n_classes)


  def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        if self.out_boundary:
            if self.boundary_with_shared:
             x_out1 = self.up1(x5)
             x_boundary_1 = self.up1_forboundary(x5, x_out1)

             x_out2 = self.up2(x_out1,x_boundary_1)
             x_boundary_2 = self.up2_forboundary(x_boundary_1,x_out2)

             x_out3 = self.up3(x_out2, x_boundary_2)
             x_boundary_3 = self.up3_forboundary(x_boundary_2, x_out3)

             x_out4 = self.up4(x_out3, x_boundary_3)
             x_boundary_4 = self.up4_forboundary(x_boundary_3, x_out4)


             x_out =   self.outc(x_out4)
             x_boundary_out =  self.outc_forboundary(x_boundary_4)

             return x_out,x_boundary_out

            else:
              x_out1 = self.up1(x5)
              x_out2 = self.up2(x_out1)
              x_out3 = self.up3(x_out2)
              x_out4 = self.up4(x_out3)
              x_out = self.outc(x_out4)

              x_boundary_1 = self.up1_forboundary(x5,x_out1)                           #x__________>________x________>________x_______>_________>
              x_boundary_2 = self.up2_forboundary(x_boundary_1, x_out2)                #            |                |               |
              x_boundary_3 = self.up3_forboundary(x_boundary_2, x_out3)                #            |                |               |
              x_boundary_4 = self.up4_forboundary(x_boundary_3, x_out4)                #x___________>_______x________>_________x_______>________>
              x_boundary_out = self.outc_forboundary(x_boundary_4)
              return x_out,x_boundary_out
        else:
              x_out1 = self.up1(x5)
              x_out2 = self.up2(x_out1)
              x_out3 = self.up3(x_out2)
              x_out4 = self.up4(x_out3)
              x_out = self.outc(x_out4)
              return x_out

def unet4_forboundary(in_channels, **kwargs):
    return UNet_forboundary(in_channels, w=4, **kwargs)

def unet4_withoutshortcut(in_channels, **kwargs):
    return UNet_forboundary_withoutshortcut(in_channels, w=4,out_boundary= True,boundary_with_shared = True, **kwargs)

def unet4_withoutshortcut_noboundary(in_channels, **kwargs):
    return UNet_forboundary_withoutshortcut(in_channels, w=4, out_boundary= False,boundary_with_shared = False, **kwargs)

def unet05(in_channels, **kwargs):
    return UNet(in_channels, w=0.5, **kwargs)

def unet025(in_channels, **kwargs):
    return UNet(in_channels, w=0.25, **kwargs)

def unet1(in_channels, **kwargs):
    return UNet(in_channels, w=1, **kwargs)

def unet2(in_channels, **kwargs):
    return UNet(in_channels, w=2, **kwargs)

def unet4(in_channels, **kwargs):
    return UNet(in_channels, w=4, **kwargs)

def unet1d2(in_channels, **kwargs):
    return UNetD2(in_channels, w=1, **kwargs)

def unet2d2(in_channels, **kwargs):
    return UNetD2(in_channels, w=2, **kwargs)

def unet4d2(in_channels, **kwargs):
    return UNetD2(in_channels, w=4, **kwargs)

def unet1d3(in_channels, **kwargs):
    return UNetD3(in_channels, w=1, **kwargs)

def unet2d3(in_channels, **kwargs):
    return UNetD3(in_channels, w=2, **kwargs)

def unet4d3(in_channels, **kwargs):
    return UNetD3(in_channels, w=4, **kwargs)

