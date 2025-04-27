import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as nn

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # 加载图像
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)

        # 创建掩码
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # 转换为张量并预处理
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            mask = nn.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        return image, mask