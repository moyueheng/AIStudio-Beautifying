from models import UNet
from data_set import MyDateset
import paddle
from loss import SSIMLoss, PSNRLoss

from visualdl import LogWriter




# 训练
model = UNet()
model.train()
# 数据集
train_dataset = MyDateset()

# # 需要接续之前的模型重复训练可以取消注释
# param_dict = paddle.load('./model.pdparams')
# model.load_dict(param_dict)

train_dataloader = paddle.io.DataLoader(
    train_dataset, batch_size=16, shuffle=True, drop_last=False
)
# 损失函数
losspsnr = PSNRLoss()
lossfn = SSIMLoss(window_size=3, data_range=1)


writer = LogWriter(logdir="./log/unet/train")


# 训练配置
max_epoch = 240
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=0.001, T_max=max_epoch
)
opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

now_step = 0
for epoch in range(max_epoch):
    for step, data in enumerate(train_dataloader):
        now_step += 1

        img, label = data
        pre = model(img)
        loss1 = lossfn(pre, label).mean()
        loss2 = losspsnr(pre, label).mean()
        loss = (loss1 + loss2 / 100) / 2

        loss.backward()
        opt.step()
        opt.clear_gradients()
        if now_step % 100 == 0:
            writer.add_scalar('loss-step', loss.mean().numpy(), now_step)
            print(
                "epoch: {}, batch: {}, loss is: {}".format(
                    epoch, step, loss.mean().numpy()
                )
            )
    if epoch % 10 == 0:
        writer.add_scalar('loss-epoch', loss, epoch)
        paddle.save(model.state_dict(), f"output/{epoch}_model.pdparams")

writer.close()

paddle.save(model.state_dict(), "final_model.pdparams")
