import paddle
import os
import cv2


# 定义数据读取器
class MyDateset(paddle.io.Dataset):
    # 构造函数，初始化数据集
    # mode: 模式，'train'或者其他
    # watermark_dir: 水印图像的路径
    # bg_dir: 背景（或者说是标签）图像的路径
    def __init__(
        self,
        mode="train",
        watermark_dir="/jinx/AIStudio/Beautifying/train_datasets/image/",
        bg_dir="/jinx/AIStudio/Beautifying/train_datasets/groundtruth/",
    ):
        # 调用父类的构造函数
        super(MyDateset, self).__init__()
        # 初始化模式
        self.mode = mode
        # 初始化水印图像的路径, 有水印的图片就是有痘痘的图片
        self.watermark_dir = watermark_dir
        # 初始化背景图像的路径, 背景图片我们可以认为就是没有痘痘的图片
        self.bg_dir = bg_dir
        # 获取水印图像的文件列表
        self.train_list = os.listdir(self.watermark_dir)

    def __getitem__(self, index):
        """获取单个元素

        Args:
            index (int): 

        Returns:
            tensor: (3,512,512) 有痘痘
            tensor: (3,512,512) 没痘痘
        """
        # 获取每一个有痘痘的图片
        item = self.train_list[index]
        # print(item)
        bg_item = self.train_list[index]
        img = cv2.imread(self.watermark_dir + item)
        # 没痘痘的图片
        label = cv2.imread(self.bg_dir + bg_item)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # 图片放缩
        img = paddle.vision.transforms.resize(
            img, (512, 512), interpolation="bilinear"
        )  # 有痘痘
        label = paddle.vision.transforms.resize(
            label, (512, 512), interpolation="bilinear"
        )  # 没痘痘
        # 对图像进行转置，使其从HWC变为CHW
        img = img.transpose((2, 0, 1))  # (3,512,512)
        label = label.transpose((2, 0, 1))  # (3,512,512)
        # 将图像的像素值归一化到[0,1]
        img = img / 255
        label = label / 255
        # # 将numpy数组转为Paddle的Tensor，并设置数据类型为float32
        img = paddle.to_tensor(img).astype("float32")
        label = paddle.to_tensor(label).astype("float32")
        return img, label

    def __len__(self):
        return len(self.train_list)
