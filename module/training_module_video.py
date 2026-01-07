import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video

from cnn_for_video import VideoClassifier


class WLASLMP4Dataset(Dataset):
    def __init__(self, root, json_file, split,
                 num_frames=8, limit_classes=10,
                 transform=None, shared_gloss_to_idx=None):
        self.root = Path(root)
        self.videos_dir = self.root / "videos"
        self.num_frames = num_frames
        self.transform = transform

        # Load class map
        class_map = {}
        with open(self.root / "wlasl_class_list.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_map[int(parts[0])] = " ".join(parts[1:])

        meta = json.loads((self.root / json_file).read_text(encoding="utf-8"))

        items = []
        for vid, info in meta.items():
            if info.get("subset") != split:
                continue
            actions = info.get("action", [])
            if not actions:
                continue
            gloss = class_map.get(actions[0])
            if gloss is None:
                continue
            mp4 = self.videos_dir / f"{vid}.mp4"
            if mp4.exists():
                items.append((str(mp4), gloss))

        if shared_gloss_to_idx is None:
            glosses = sorted({g for _, g in items})[:limit_classes]
            self.gloss_to_idx = {g: i for i, g in enumerate(glosses)}
        else:
            self.gloss_to_idx = shared_gloss_to_idx

        self.samples = [
            (p, self.gloss_to_idx[g])
            for p, g in items
            if g in self.gloss_to_idx
        ]

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, n):
        if n <= self.num_frames:
            return torch.linspace(0, n - 1, steps=self.num_frames).long()
        start = (n - self.num_frames) // 2
        return torch.arange(start, start + self.num_frames)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video, _, _ = read_video(path, pts_unit="sec")
        if video.numel() == 0:
            video = torch.zeros((self.num_frames, 224, 224, 3), dtype=torch.uint8)

        n = video.shape[0]
        idxs = self._sample_indices(n).clamp(0, max(n - 1, 0))
        clip = video[idxs].permute(0, 3, 1, 2)

        if self.transform:
            clip = torch.stack([self.transform(f) for f in clip])

        return clip, label


def main():
    ROOT = r"D:\hand-sign-detector-using-2-mixure-cnn-for-picture-and-video\data\WLASL"
    JSON_FILE = "nslt_100.json"

    NUM_FRAMES = 8
    LIMIT_CLASSES = 10
    EPOCHS = 20
    BATCH_SIZE = 4
    TEMPORAL_MODE = "tcn"
    LR = 1e-3
    PATIENCE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    train_ds = WLASLMP4Dataset(
        ROOT, JSON_FILE, "train",
        num_frames=NUM_FRAMES,
        limit_classes=LIMIT_CLASSES,
        transform=tf
    )

    val_ds = WLASLMP4Dataset(
        ROOT, JSON_FILE, "test",
        num_frames=NUM_FRAMES,
        limit_classes=LIMIT_CLASSES,
        transform=tf,
        shared_gloss_to_idx=train_ds.gloss_to_idx
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = VideoClassifier(
        num_classes=len(train_ds.gloss_to_idx),
        temporal_mode=TEMPORAL_MODE
    ).to(device)

    for name, param in model.named_parameters():
        if "cnn" in name or "backbone" in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    best_acc = 0.0
    bad_epochs = 0

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = total = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / max(1, total)

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate()
        print(f"Epoch {ep:02d}/{EPOCHS} | val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    print(f"Best val_acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
