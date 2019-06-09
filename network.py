"""
This implementation is based on End to End Image compression

"""

from torch import nn

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x



class End_to_End(nn.Module):
    def __init__(self):
        super(End_to_End, self).__init__()

        # Encoder
        # TODO : try with padding = 0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Decoder
        # TODO : try ConvTranspose2d
        self.interpolate = Interpolate(size=96, mode='bilinear')
        self.deconv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)

        self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)

        self.deconv3 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn1(out)
        return self.conv3(out)

    def reparameterize(self, mu, logvar):
        pass

    def decode(self, z):
        upscaled_image = self.interpolate(z)
        out = self.relu(self.deconv1(upscaled_image))
        out = self.relu(self.deconv2(out))
        out = self.bn2(out)
        for _ in range(5):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        out = self.deconv3(out)
        final = upscaled_image + out
        return final, out, upscaled_image

    def forward(self, x):
        com_img = self.encode(x)
        final, out, upscaled_image = self.decode(com_img)
        return final, out, upscaled_image, com_img, x