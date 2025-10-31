import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import build_config
from torchvision import transforms
from PIL import Image

class UCFFrameDataset(Dataset):
    def __init__(self, list_file, transform=None):
        with open(list_file, 'r') as f:
            self.frame_dirs = [line.strip() for line in f if line.strip()]
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.max_frames = 256  # desired fixed length

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, idx):
        folder_path = self.frame_dirs[idx]
        frame_files = sorted([
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith('.jpg')
        ])

        frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path).convert("RGB")
            img_tensor = self.transform(img)  # [3, 224, 224]
            frames.append(img_tensor)

        video_tensor = torch.stack(frames)  # [T, 3, 224, 224]
        num_frames = video_tensor.shape[0]

        # Zero-pad if fewer than max_frames
        if num_frames < self.max_frames:
            pad_size = self.max_frames - num_frames
            pad_tensor = torch.zeros((pad_size, 3, 224, 224))
            video_tensor = torch.cat([video_tensor, pad_tensor], dim=0)
        else:
            # optionally truncate if longer
            video_tensor = video_tensor[:self.max_frames]

        label = 0 if "Normal" in folder_path else 1

        return video_tensor, num_frames, label, folder_path


def load_data(training = True):
    cfg = build_config()

    train_dataset = UCFFrameDataset(cfg.train_set)
    val_dataset = UCFFrameDataset(cfg.val_set)
    test_dataset = UCFFrameDataset(cfg.test_set)

    if training:
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.val_bs, shuffle=False)
        return train_loader, val_loader
    else:
        val_loader = DataLoader(val_dataset, batch_size=cfg.val_bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_bs, shuffle=False)
        return val_loader, test_loader
    
