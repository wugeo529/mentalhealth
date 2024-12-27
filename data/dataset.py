import os
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset
from PIL import Image

class SpectrogramDataset(Dataset):
    def __init__(self, label_file, spectrogram_root, transform=None, device="cpu"):
        self.spectrogram_root = spectrogram_root
        self.transform = transform
        self.device = device

        # 加载并过滤标签
        labels = pd.read_csv(label_file)
        valid_rows = []
        for _, row in labels.iterrows():
            case_id = str(row['Case_id'])
            case_folder = os.path.join(self.spectrogram_root, case_id)
            if os.path.exists(os.path.join(case_folder, "1.png")) and os.path.exists(os.path.join(case_folder, "2.png")):
                valid_rows.append(row)
        self.labels = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        case_id = str(self.labels.iloc[idx]['Case_id'])
        label = int(self.labels.iloc[idx]['Label']) - 1  # 确保从 0 开始

        # 加载图像
        case_folder = os.path.join(self.spectrogram_root, case_id)
        spec1_path = os.path.join(case_folder, "1.png")
        spec2_path = os.path.join(case_folder, "2.png")

        spec1 = Image.open(spec1_path).convert("RGB")
        spec2 = Image.open(spec2_path).convert("RGB")

        if self.transform:
            spec1 = self.transform(spec1)
            spec2 = self.transform(spec2)

        # 使用 ResNet 提取特征
        resnet = resnet18(pretrained=True).to(self.device)
        resnet.fc = torch.nn.Identity()  # 移除最后分类层
        with torch.no_grad():
            spec1_features = resnet(spec1.unsqueeze(0).to(self.device)).squeeze(0)
            spec2_features = resnet(spec2.unsqueeze(0).to(self.device)).squeeze(0)

        return torch.stack([spec1_features, spec2_features], dim=0), label
