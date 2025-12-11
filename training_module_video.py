import os
import json
import random
import argparse
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision import transforms
from cnn_for_video import DenseNetWithTemporalResidual


def load_wlasl(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    glosses = sorted({entry["gloss"] for entry in data})
    gloss2id = {g: i for i, g in enumerate(glosses)}

    splits = {"train": [], "val": [], "test": []}

    for entry in data:
        gloss = entry["gloss"]
        label = gloss2id[gloss]

        for inst in entry["instances"]:
            video_id = inst["video_id"]
            split = inst.get("split", "train")

            if split not in splits:
                split = "train"

            splits[split].append({
                "video_id": video_id,
                "label": label,
                "gloss": gloss
            })

    print(f"Loaded WLASL: {len(gloss2id)} classes")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")

    return splits, gloss2id



class WLASLDataset(Dataset):

    def __init__(self, samples: List[dict], video_root: str,
                 num_frames: int = 16, train=True):

        self.samples = samples
        self.video_root = video_root
        self.num_frames = num_frames
        self.train = train

        self.frame_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def _path(self, video_id):
        return os.path.join(self.video_root, f"{video_id}.mp4")

    def _sample_indices(self, F):
        if F <= self.num_frames:
            idx = list(range(F))
            while len(idx) < self.num_frames:
                idx.append(F - 1)
            return idx

        if self.train:
            start = random.randint(0, F - self.num_frames)
        else:
            start = (F - self.num_frames) // 2

        return list(range(start, start + self.num_frames))

    def __getitem__(self, idx):
        info = self.samples[idx]
        vid = info["video_id"]
        label = info["label"]

        p = self._path(vid)
        if not os.path.exists(p):
            raise FileNotFoundError(p)

        video, _, _ = read_video(p, pts_unit="sec")  # (T, H, W, C)
        T_total = video.shape[0]

        idxs = self._sample_indices(T_total)

        frames = []
        for t in idxs:
            frame = video[t]
            frame = self.frame_tf(frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)

        return frames, label

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += frames.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss = criterion(logits, labels)

        total_loss += loss.item() * frames.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += frames.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser("Train WLASL video model")

    parser.add_argument("--json", type=str, default="WLASL_v0.3.json")
    parser.add_argument("--video-root", type=str, default="videos",
                        help="Thư mục chứa video_id.mp4")

    parser.add_argument("--epochs", type=int, default=12)  
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--save", type=str, default="best_wlasl.pth")

    # Chặn lỗi SystemExit khi debug trong VSCode
    try:
        args = parser.parse_args()
    except SystemExit:
        print("\n⚠ Argparse không nhận được tham số. Dùng DEFAULT cho VSCode Debug.\n")
        class Args:
            json = "WLASL_v0.3.json"
            video_root = "videos"
            epochs = 12
            batch_size = 4
            lr = 1e-4
            num_frames = 16
            save = "debug_wlasl.pth"

        args = Args()

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load annotation
    splits, gloss2id = load_wlasl(args.json)
    num_classes = len(gloss2id)

    # 2) Dataset
    train_ds = WLASLDataset(
        splits["train"], args.video_root,
        args.num_frames, train=True
    )
    val_ds = WLASLDataset(
        splits["val"], args.video_root,
        args.num_frames, train=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # 3) Model
    model = DenseNetWithTemporalResidual(
        num_classes=num_classes,
        temporal_mode="tcn",
        temporal_pool="avg"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0

    # 4) Train loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "gloss2id": gloss2id
            }, args.save)

            print(f"  → Saved BEST model to {args.save} (Val Acc={val_acc:.4f})")

    print("\nTraining completed. Best Val Accuracy:", best_val_acc)


if __name__ == "__main__":
    main()
