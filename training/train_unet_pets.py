# %% pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# %% pip install torchmetrics onnx onnxsim onnx2tf tensorflow==2.13.1 Pillow -q

import os, numpy as np
from pathlib import Path

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import OxfordIIITPet

import torchmetrics
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 8))
LR = float(os.environ.get("LR", 1e-3))
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
SAVE_BEST = "training/unet_best.pt"

# --------- transforms ----------
img_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
mask_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
    T.PILToTensor(),  # uint8 [1,H,W] com {1,2,3}
])

class PetsSegBin(torch.utils.data.Dataset):
    def __init__(self, split="trainval", indices=None):
        self.ds = OxfordIIITPet(root=DATA_ROOT, split=split, target_types="segmentation", download=True)
        self.indices = list(range(len(self.ds))) if indices is None else list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, mask = self.ds[self.indices[i]]
        img = img_tf(img)
        mask = mask_tf(mask).squeeze(0)  # [H,W]
        mask = ((mask==1) | (mask==2)).float().unsqueeze(0)  # [1,H,W]
        return img, mask

# recorte p/ acelerar
full = PetsSegBin("trainval")
N = len(full)
g = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=g).tolist()
train_idx = perm[:1000]
val_idx   = perm[1000:1300]

ds_tr = PetsSegBin("trainval", train_idx)
ds_va = PetsSegBin("trainval", val_idx)

dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --------- U-Net compacta ----------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base=32, out_channels=1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)
        b = self.bottleneck(p4)
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1)  # logits

def train():
    model = UNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    iou_metric = torchmetrics.JaccardIndex(task="binary").to(DEVICE)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(DEVICE)

    best_iou = -1.0
    for ep in range(1, EPOCHS+1):
        # train
        model.train(); iou_metric.reset(); acc_metric.reset()
        tl, tn = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                iou_metric.update(preds, y.int())
                acc_metric.update(preds, y.int())
            tl += loss.item()*x.size(0); tn += x.size(0)
        tiou, tacc = iou_metric.compute().item(), acc_metric.compute().item()

        # val
        model.eval(); iou_metric.reset(); acc_metric.reset()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(x)
                loss = crit(logits, y)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                iou_metric.update(preds, y.int())
                acc_metric.update(preds, y.int())
                vl += loss.item()*x.size(0); vn += x.size(0)
        viou, vacc = iou_metric.compute().item(), acc_metric.compute().item()

        print(f"Ep {ep:02d} | train loss {tl/tn:.4f} IoU {tiou:.3f} acc {tacc:.3f} || "
              f"val loss {vl/vn:.4f} IoU {viou:.3f} acc {vacc:.3f}")

        if viou > best_iou:
            best_iou = viou
            Path(SAVE_BEST).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_BEST)

    print("Melhor IoU (val):", best_iou)
    return model

@torch.no_grad()
def save_examples(checkpoint=SAVE_BEST, outdir="training/samples", n=4):
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    os.makedirs(outdir, exist_ok=True)
    batch = next(iter(dl_va))
    x, y = batch
    x = x.to(DEVICE)
    preds = (torch.sigmoid(model(x)) >= 0.5).float().cpu()
    # desfaz normalização p/ visualização
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    for i in range(min(n, x.size(0))):
        img = (x[i].cpu()*std + mean).clamp(0,1)
        base = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
        m = preds[i,0].numpy() > 0.5
        overlay = base.copy()
        overlay[m] = [255,0,0]
        blended = (0.6*base + 0.4*overlay).astype(np.uint8)
        Image.fromarray(blended).save(f"{outdir}/pred_{i}.png")

if __name__ == "__main__":
    model = train()
    save_examples()
    print("Exemplos salvos em training/samples/")
