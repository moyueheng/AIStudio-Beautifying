import paddle

# Generator Code


class UnetGenerator(paddle.nn.Layer):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator, self).__init__()

        self.down1 = paddle.nn.Conv2D(
            input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.down2 = Downsample(ngf, ngf*2)
        self.down3 = Downsample(ngf*2, ngf*4)
        self.down4 = Downsample(ngf*4, ngf*8)
        self.down5 = Downsample(ngf*8, ngf*8)
        self.down6 = Downsample(ngf*8, ngf*8)
        self.down7 = Downsample(ngf*8, ngf*8)

        self.center = Downsample(ngf*8, ngf*8)

        self.up7 = Upsample(ngf*8, ngf*8, use_dropout=True)
        self.up6 = Upsample(ngf*8*2, ngf*8, use_dropout=True)
        self.up5 = Upsample(ngf*8*2, ngf*8, use_dropout=True)
        self.up4 = Upsample(ngf*8*2, ngf*8)
        self.up3 = Upsample(ngf*8*2, ngf*4)
        self.up2 = Upsample(ngf*4*2, ngf*2)
        self.up1 = Upsample(ngf*2*2, ngf)

        self.output_block = paddle.nn.Sequential(
            paddle.nn.ReLU(),
            paddle.nn.Conv2DTranspose(
                ngf*2, output_nc, kernel_size=4, stride=2, padding=1),
            paddle.nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        c = self.center(d7)

        x = self.up7(c, d7)
        x = self.up6(x, d6)
        x = self.up5(x, d5)
        x = self.up4(x, d4)
        x = self.up3(x, d3)
        x = self.up2(x, d2)
        x = self.up1(x, d1)

        x = self.output_block(x)
        return x


class Downsample(paddle.nn.Layer):
    # LeakyReLU => conv => batch norm
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(Downsample, self).__init__()

        self.layers = paddle.nn.Sequential(
            paddle.nn.LeakyReLU(0.2),
            paddle.nn.Conv2D(in_dim, out_dim, kernel_size,
                             stride, padding, bias_attr=False),
            paddle.nn.BatchNorm2D(out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Upsample(paddle.nn.Layer):
    # ReLU => deconv => batch norm => dropout
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(Upsample, self).__init__()

        sequence = [
            paddle.nn.ReLU(),
            paddle.nn.Conv2DTranspose(
                in_dim, out_dim, kernel_size, stride, padding, bias_attr=False),
            paddle.nn.BatchNorm2D(out_dim)
        ]

        if use_dropout:
            sequence.append(paddle.nn.Dropout(p=0.5))

        self.layers = paddle.nn.Sequential(*sequence)

    def forward(self, x, skip):
        x = self.layers(x)
        x = paddle.concat([x, skip], axis=1)
        return x


if __name__ == "__main__":

    # 通过paddle.summary可以查看一个指定形状的数据在网络中各个模块中的传递
    paddle.summary(UnetGenerator(), (8, 3, 512, 512))
