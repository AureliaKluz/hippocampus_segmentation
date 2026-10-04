import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.utils import set_determinism

from hippo_data import get_dataset

set_determinism(seed=42)
device = torch.device("cpu")

ds = get_dataset("train")
items = [ds[0], ds[1]]
img = torch.stack([it["image"].as_tensor() for it in items]).to(device)
lbl = torch.stack([it["label"].as_tensor() for it in items]).to(device)
print("Przypadki:", [it["case"] for it in items])
print("Obraz:", tuple(img.shape), "Maska:", tuple(lbl.shape))

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=1,
).to(device)
print("Liczba parametrów:", sum(p.numel() for p in model.parameters()))

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def dice_per_label(logits, target):
    pred = logits.argmax(dim=1)
    target = target.squeeze(1).long()
    out = {}
    for c, name in [(1, "Anterior"), (2, "Posterior")]:
        inter = ((pred == c) & (target == c)).sum().item()
        denom = (pred == c).sum().item() + (target == c).sum().item()
        out[name] = 2 * inter / max(denom, 1)
    return out


EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(img)
    loss = loss_fn(logits, lbl)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        with torch.no_grad():
            d = dice_per_label(logits, lbl)
        print(f"epoka {epoch:3d} | loss {loss.item():.4f} | "
              f"Dice Anterior {d['Anterior']:.3f} | "
              f"Dice Posterior {d['Posterior']:.3f}")

model.eval()
with torch.no_grad():
    pred = model(img).argmax(dim=1)
for i, it in enumerate(items):
    for c, name in [(1, "Anterior"), (2, "Posterior")]:
        n_pred = int((pred[i] == c).sum())
        n_true = int((lbl[i, 0] == c).sum())
        print(f"{it['case']} {name}: przewidziane {n_pred} mm3, "
              f"referencyjne {n_true} mm3")