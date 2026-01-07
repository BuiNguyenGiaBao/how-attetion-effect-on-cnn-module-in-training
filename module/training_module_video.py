from __future__ import annotations
import argparse, random, time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def apply_video_transform(video: torch.Tensor, img_size: int):
    if video.dtype == torch.uint8:
        video = video.float() / 255.0
    video = F.interpolate(video, size=(img_size, img_size),
                          mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=video.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=video.device).view(1,3,1,1)
    return (video - mean) / std


class UCF101Clips(Dataset):
    def __init__(self, base_ds, img_size: int):
        self.base = base_ds
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        video = item[0]
        label = item[2] if len(item) >= 3 else item[1]
        video = apply_video_transform(video, self.img_size)
        return video, int(label)


class RemapLabels(Dataset):
    def __init__(self, base, remap: Dict[int, int]):
        self.base = base
        self.remap = remap

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, y = self.base[idx]
        return video, self.remap.get(int(y), int(y))


def build_ucf101(args):
    base_train = datasets.UCF101(
        root=args.data_root,
        annotation_path=args.ann_root,
        frames_per_clip=args.num_frames,
        step_between_clips=args.step_between_clips,
        train=True,
        fold=args.fold,
        num_workers=args.decode_workers,
    )
    base_val = datasets.UCF101(
        root=args.data_root,
        annotation_path=args.ann_root,
        frames_per_clip=args.num_frames,
        step_between_clips=args.step_between_clips,
        train=False,
        fold=args.fold,
        num_workers=args.decode_workers,
    )

    classes = base_train.classes
    subset = classes[:args.subset_n] if args.subset_n > 0 else classes
    class_to_idx = base_train.class_to_idx
    keep = [class_to_idx[c] for c in subset]

    train_idx = [i for i in range(len(base_train)) if int(base_train[i][2]) in keep]
    val_idx   = [i for i in range(len(base_val)) if int(base_val[i][2]) in keep]

    base_train = Subset(base_train, train_idx)
    base_val   = Subset(base_val, val_idx)

    remap = {old: new for new, old in enumerate(keep)}
    train_ds = RemapLabels(UCF101Clips(base_train, args.img_size), remap)
    val_ds   = RemapLabels(UCF101Clips(base_val, args.img_size), remap)

    return train_ds, val_ds, subset


def build_model(args):
    from cnn_for_video import VideoClassifier
    return VideoClassifier(num_classes=args.num_classes, temporal_mode=args.temporal_mode)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = loss_sum = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ann_root", required=True)
    p.add_argument("--subset_n", type=int, default=5)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--step_between_clips", type=int, default=8)
    p.add_argument("--decode_workers", type=int, default=2)
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temporal_mode", default="tcn")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root", default="D:/hand-sign-detector-using-2-mixure-cnn-for-picture-and-video/data")
    p.add_argument("--ann_root", default="D:/datasets/ucfTrainTestlist")

    args = p.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    train_ds, val_ds, classes = build_ucf101(args)
    args.num_classes = len(classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, drop_last=False)

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()

        _, acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {ep:03d} | val_acc={acc:.4f}")
        best = max(best, acc)

    print(f"Best val_acc={best:.4f}")


if __name__ == "__main__":
    main()
