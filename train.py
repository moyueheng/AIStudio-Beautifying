import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from models import NLayerDiscriminator, UnetGenerator


# dataset
from data_set import MyDateset
dataset = MyDateset()
dataloader = paddle.io.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False)

# create Net
netG = UnetGenerator()
netD = NLayerDiscriminator()

# 如果想要接着之前训练的模型训练，将if 0修改为if 1即可
if 0:
    try:
        mydict = paddle.load('generator.params')
        netG.set_dict(mydict)
        mydict = paddle.load('discriminator.params')
        netD.set_dict(mydict)
    except:
        print('fail to load model')

netG.train()
netD.train()
# optimizer配置
optimizerD = paddle.optimizer.Adam(
    parameters=netD.parameters(), learning_rate=0.00002, beta1=0.5, beta2=0.999)
optimizerG = paddle.optimizer.Adam(
    parameters=netG.parameters(), learning_rate=0.00002, beta1=0.5, beta2=0.999)

# Loss配置
bce_loss = paddle.nn.BCELoss()
l1_loss = paddle.nn.L1Loss()


# 最大迭代epoch
max_epoch = 240

now_step = 0
for epoch in range(max_epoch):
    for step, (img, mask) in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # 清除D的梯度
        optimizerD.clear_grad()

        # 传入正样本，并更新梯度, 让判别器判断正样本是真的还是假的
        pos_img = paddle.concat((img, mask), 1)  # img是没有痘痘的, mask是有痘痘的,
        # label = paddle.full([pos_img.shape[0], 1], 1, dtype='float32') # 判别器判断出来的类别是什么
        pre = netD(pos_img)
        loss_D_1 = bce_loss(pre, paddle.ones_like(pre))
        loss_D_1.backward()

        # 通过randn构造随机数，制造负样本，并传入D，更新梯度
        fake_img = netG(mask).detach()
        neg_img = paddle.concat((fake_img, mask), 1)
        # label = paddle.full([pos_img.shape[0], 1], 0, dtype='float32')
        pre = netD(neg_img.detach())  # 通过detach阻断网络梯度传播，不影响G的梯度计算
        loss_D_2 = bce_loss(pre, paddle.zeros_like(pre))
        loss_D_2.backward()

        # 更新D网络参数
        optimizerD.step()
        optimizerD.clear_grad()

        loss_D = loss_D_1 + loss_D_2

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        # 清除D的梯度
        optimizerG.clear_grad()

        fake_img = netG(mask)
        fake = paddle.concat((fake_img, mask), 1)
        # label = paddle.full((pos_img.shape[0], 1), 1, dtype=np.float32,)
        output = netD(fake)
        loss_G_1 = l1_loss(fake_img, img) * 100.
        loss_G_2 = bce_loss(output, paddle.ones_like(pre))
        loss_G = loss_G_1+loss_G_2
        loss_G.backward()

        # 更新G网络参数
        optimizerG.step()
        optimizerG.clear_grad()

        now_step += 1

        print('\r now_step is:', now_step, end='')
        ###########################
        # 可视化
        ###########################
        if now_step % 100 == 0:
            print()
            try:
                for i in range(3):
                    # image = generated_image[i].transpose()
                    image = fake_img[i]
                    image = np.where(image > 0, image, 0)
                    image = image.transpose((1, 2, 0))
                    image = image * 255
                    image = image.astype(np.uint8)
                    cv2.imwrite('./images/' + str(epoch) + '_' + str(now_step) + '_' + str(i) + '.jpg', image)
                msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(
                    epoch, now_step, loss_D.numpy()[0], loss_G.numpy()[0])
                print(msg)
            except IOError:
                print(IOError)

    if epoch % 5 == 0:
        paddle.save(netG.state_dict(), f"./output/pix2pix/{epoch}_generator.params")
        paddle.save(netD.state_dict(), f"./output/pix2pix/{epoch}_discriminator.params")

paddle.save(netG.state_dict(), "generator.params")
paddle.save(netD.state_dict(), "discriminator.params")
