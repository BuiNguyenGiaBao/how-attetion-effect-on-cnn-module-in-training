from __future__ import annotations
import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torchvision import datasets, transforms
except Exception as e:
    datasets = None
    transforms = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp: bool) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == targets).sum().item())
        total_n += bs
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def build_transforms(img_size: int, data: str):
    if transforms is None:
        raise RuntimeError("torchvision is required. Install torchvision first.")

    data = data.lower()
    if data in ("cifar10", "cifar100"):
        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        val_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_tfms = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return train_tfms, val_tfms

#check if data is in the datas folder is not it will download from website 
#you can add your data in imagefolder if using differnt dataset use it  in if else loop below
def build_dataloaders(data: str, data_dir: str, batch_size: int, num_workers: int, img_size: int):
    if datasets is None:
        raise RuntimeError("torchvision is required. Install torchvision first.")

    train_tfms, val_tfms = build_transforms(img_size=img_size, data=data)
    data = data.lower()

    if data == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
        val_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_tfms)
    elif data == "cifar100":
        train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tfms)
        val_ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_tfms)
    elif data == "imagefolder":
        train_root = os.path.join(data_dir, "train")
        val_root = os.path.join(data_dir, "val")
        if not (os.path.isdir(train_root) and os.path.isdir(val_root)):
            raise FileNotFoundError(
                "For --data imagefolder, expected:\n"
                f"  {train_root}/<class_name>/*\n"
                f"  {val_root}/<class_name>/*"
            )
        train_ds = datasets.ImageFolder(train_root, transform=train_tfms)
        val_ds = datasets.ImageFolder(val_root, transform=val_tfms)
    else:
        raise ValueError("--data must be: cifar10 | cifar100 | imagefolder")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(args) -> nn.Module:
    from cnn_for_pic import DenseNetWithLinearAttentionn

    return DenseNetWithLinearAttentionn(
        num_classes=args.num_classes,
        growth=args.growth,
        bn_size=args.bn_size,
        drop_rate=args.drop_rate,
        num_layers_block1=args.layers_b1,
        num_layers_block2=args.layers_b2,
        d_k=args.d_k,
        d_v=args.d_v,
        use_gn_head=args.use_gn_head,
        num_groups=args.gn_groups,
    )


def build_criterion(args) -> nn.Module:
    if args.loss == "ce":
        if args.label_smoothing > 0:
            try:
                return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            except TypeError:
                return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss()
    if args.loss == "focal":
        from cnn_for_pic import FocalLoss
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    raise ValueError("--loss must be: ce | focal")


@dataclass
class RunState:
    epoch: int = 0
    best_acc: float = 0.0
    global_step: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagefolder"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--img_size", type=int, default=224)

    # Model
    p.add_argument("--num_classes", type=int, default=100) #quirck remind if you use another dataset remmember to change the lable for that data
    p.add_argument("--growth", type=int, default=24)
    p.add_argument("--bn_size", type=int, default=2)
    p.add_argument("--drop_rate", type=float, default=0.2)
    p.add_argument("--layers_b1", type=int, default=3)
    p.add_argument("--layers_b2", type=int, default=3)
    p.add_argument("--d_k", type=int, default=64)
    p.add_argument("--d_v", type=int, default=64)
    p.add_argument("--use_gn_head", action="store_true")
    p.add_argument("--gn_groups", type=int, default=8)

    # Train
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)

    # Loss
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_alpha", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # Scheduler
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--step_size", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.1)

    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")

    # IO
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_every", type=int, default=0)

    return p.parse_args()


def save_ckpt(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = build_dataloaders(
        data=args.data,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    model = build_model(args).to(device)
    criterion = build_criterion(args)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    if args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    state = RunState()

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optim"])
        if scheduler is not None and ckpt.get("sched") is not None:
            scheduler.load_state_dict(ckpt["sched"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        st = ckpt.get("state", {})
        state.epoch = int(st.get("epoch", 0))
        state.best_acc = float(st.get("best_acc", 0.0))
        state.global_step = int(st.get("global_step", 0))
        print(f"[Resume] epoch={state.epoch} best_acc={state.best_acc:.4f}")

    print(f"[Run] out_dir={out_dir}")
    print(f"[Device] {device} | amp={args.amp and device.type == 'cuda'}")

    for epoch in range(state.epoch, args.epochs):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == targets).sum().item())
            total_n += bs
            state.global_step += 1

        train_loss = total_loss / max(1, total_n)
        train_acc = total_correct / max(1, total_n)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.amp)
        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | lr={lr:.3e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s"
        )

        payload = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": asdict(state),
            "args": vars(args),
        }

        save_ckpt(out_dir / "last.pt", payload)
        if val_acc > state.best_acc:
            state.best_acc = val_acc
            save_ckpt(out_dir / "best.pt", payload)
            print(f"  -> New best: {state.best_acc:.4f} (saved best.pt)")

        if args.save_every and ((epoch + 1) % args.save_every == 0):
            save_ckpt(out_dir / f"epoch_{epoch+1:03d}.pt", payload)

        state.epoch = epoch + 1

    print(f"Done. Best val_acc={state.best_acc:.4f}. Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
