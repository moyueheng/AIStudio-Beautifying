import paddle

# Discriminator Code
class NLayerDiscriminator(paddle.nn.Layer):
    def __init__(self, input_nc=6, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        self.layers = paddle.nn.Sequential(
            paddle.nn.Conv2D(input_nc, ndf, kernel_size=4, stride=2, padding=1), 
            paddle.nn.LeakyReLU(0.2),
            
            ConvBlock(ndf, ndf*2),
            ConvBlock(ndf*2, ndf*4),
            ConvBlock(ndf*4, ndf*8, stride=1),

            paddle.nn.Conv2D(ndf*8, 1, kernel_size=4, stride=1, padding=1),
            paddle.nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)


class ConvBlock(paddle.nn.Layer):
    # conv => batch norm => LeakyReLU
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.layers = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_dim, out_dim, kernel_size, stride, padding, bias_attr=False),
            paddle.nn.BatchNorm2D(out_dim),
            paddle.nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
    
if __name__ == "__main__":

    # 通过paddle.summary可以查看一个指定形状的数据在网络中各个模块中的传递
    paddle.summary(NLayerDiscriminator(), (16, 6, 512, 512))
