import argparse
import csv
import json
import time

import monai
import torch
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from torch.utils.data import Subset

from hippo_data import PROJECT_ROOT, get_dataset
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="nazwa uruchomienia")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--channels", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256])
parser.add_argument("--res-units", type=int, default=2)
parser.add_argument("--arch", default="unet", choices=["unet", "unetr"])
parser.add_argument("--feature-size", type=int, default=16)
parser.add_argument("--hidden-size", type=int, default=768)
parser.add_argument("--mlp-dim", type=int, default=3072)
parser.add_argument("--num-heads", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--limit", type=int, default=0,
                    help="użyj tylko N pierwszych przypadków (test lokalny)")
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--cache", action="store_true",
                    help="trzymaj przetworzone dane w pamięci RAM")
parser.add_argument("--augment", action="store_true",
                    help="augmentacje danych treningowych")
args = parser.parse_args()

set_determinism(seed=args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = PROJECT_ROOT / "results" / "runs" / args.name
run_dir.mkdir(parents=True, exist_ok=True)

config = {
    **vars(args),
    "device": str(device),
    "torch": torch.__version__,
    "monai": monai.__version__,
}
with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print(config)


def make_loader(split, shuffle):
    ds = get_dataset(split, cache=args.cache,
                     augment=args.augment and split == "train")
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers)


train_loader = make_loader("train", shuffle=True)
val_loader = make_loader("val", shuffle=False)

model = build_model(vars(args)).to(device)
print("Liczba parametrów:", sum(p.numel() for p in model.parameters()))

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)

with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        ["epoch", "train_loss", "val_dice_anterior", "val_dice_posterior",
         "val_dice_mean", "seconds"])

best = -1.0
for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    model.train()
    losses = []
    for batch in train_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_loss = sum(losses) / len(losses)

    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            preds = [post_pred(p) for p in decollate_batch(logits)]
            labels = [post_label(l) for l in decollate_batch(y)]
            dice_metric(y_pred=preds, y=labels)
    d = dice_metric.aggregate()
    d_ant, d_post = d[0].item(), d[1].item()
    d_mean = (d_ant + d_post) / 2
    seconds = time.time() - t0

    with open(run_dir / "log.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [epoch, f"{train_loss:.4f}", f"{d_ant:.4f}", f"{d_post:.4f}",
             f"{d_mean:.4f}", f"{seconds:.1f}"])

    torch.save(model.state_dict(), run_dir / "last.pt")
    mark = ""
    if d_mean > best:
        best = d_mean
        torch.save(model.state_dict(), run_dir / "best.pt")
        mark = "  <- najlepszy"
    print(f"epoka {epoch:3d} | loss {train_loss:.4f} | "
          f"val Dice Ant {d_ant:.3f} Post {d_post:.3f} "
          f"(śr. {d_mean:.3f}) | {seconds:.0f} s{mark}")

print(f"\nNajlepszy średni Dice walidacyjny: {best:.4f}")