# # src/baseline_cls.py
# import torch, torch.nn as nn
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# BATCH=32; EPOCHS=5; LR=1e-3; IMG=224; NUM_CLASSES=5  # adjust classes

# tfm_train = transforms.Compose([
#   transforms.RandomHorizontalFlip(),
#   transforms.RandomRotation(10),
#   transforms.ToTensor(),
#   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])
# tfm_eval = transforms.Compose([
#   transforms.ToTensor(),
#   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# train_ds = datasets.ImageFolder("data/train", tfm_train)
# val_ds   = datasets.ImageFolder("data/val", tfm_eval)
# train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
# val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
# net = net.to(device)

# crit = nn.CrossEntropyLoss()
# opt  = torch.optim.AdamW(net.parameters(), lr=LR)

# def run_epoch(dl, train=True):
#     net.train(train)
#     total, correct, loss_sum = 0, 0, 0.0
#     for x,y in tqdm(dl, leave=False):
#         x,y = x.to(device), y.to(device)
#         if train: opt.zero_grad()
#         p = net(x)
#         loss = crit(p,y)
#         if train:
#             loss.backward(); opt.step()
#         loss_sum += loss.item()*y.size(0)
#         correct += (p.argmax(1)==y).sum().item()
#         total += y.size(0)
#     return loss_sum/total, correct/total

# best=0
# for epoch in range(EPOCHS):
#     tr_loss,tr_acc = run_epoch(train_dl, True)
#     va_loss,va_acc = run_epoch(val_dl, False)
#     print(f"ep{epoch}: train {tr_acc:.3f} val {va_acc:.3f}")
#     if va_acc>best:
#         best=va_acc
#         torch.save(net.state_dict(), "outputs/models/baseline_cls.pt")






# src/baseline_cls.py
import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- This function can stay in the global scope ---
def run_epoch(net, dl, crit, opt, device, train=True):
    net.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in tqdm(dl, leave=False):
        x,y = x.to(device), y.to(device)
        if train: opt.zero_grad()
        p = net(x)
        loss = crit(p,y)
        if train:
            loss.backward(); opt.step()
        loss_sum += loss.item()*y.size(0)
        correct += (p.argmax(1)==y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

# --- Main execution block ---
if __name__ == '__main__':
    BATCH=32; EPOCHS=5; LR=1e-3; IMG=224; NUM_CLASSES=5  # adjust classes

    tfm_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tfm_eval = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder("data/train", tfm_train)
    val_ds   = datasets.ImageFolder("data/val", tfm_eval)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A small correction for clarity: pass net and other objects to run_epoch
    # as arguments instead of relying on them as globals.
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    net = net.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(net.parameters(), lr=LR)
    
    best=0
    for epoch in range(EPOCHS):
        tr_loss,tr_acc = run_epoch(net, train_dl, crit, opt, device, True)
        # For validation, we don't need the optimizer
        with torch.no_grad():
             va_loss,va_acc = run_epoch(net, val_dl, crit, opt, device, False)
        print(f"ep{epoch}: train {tr_acc:.3f} val {va_acc:.3f}")
        if va_acc>best:
            best=va_acc
            torch.save(net.state_dict(), "outputs/models/baseline_cls.pt")