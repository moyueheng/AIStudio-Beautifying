import paddle
import os
from os.path import join as pjoin
import cv2


class MyDateset(paddle.io.Dataset):
    def __init__(self, root_dir='/jinx/AIStudio/Beautifying/train_datasets'):
        super(MyDateset, self).__init__()

        self.root_dir = root_dir
        # 要根据mask生成img
        self.img_dir = pjoin(root_dir, 'groundtruth')  # 没痘痘的
        self.mask_dir = pjoin(root_dir, 'image')  # 有痘痘的
        self.file_list = os.listdir(self.img_dir)

    def __getitem__(self, index):

        img_path = pjoin(self.img_dir, self.file_list[index])
        mask_path = pjoin(self.mask_dir, self.file_list[index])

        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))  # （宽，高）
        img = img/255
        img = img.transpose([2, 0, 1])
        img = paddle.to_tensor(img).astype('float32')

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (512, 512))  # （宽，高）
        mask = mask/255
        mask = mask.transpose([2, 0, 1])
        mask = paddle.to_tensor(mask).astype('float32')

        return img, mask

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dataset = MyDateset()
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        drop_last=False)

    for step, data in enumerate(dataloader):
        img, mask = data
        print(step, img.shape, mask.shape)
        break
