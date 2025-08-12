from config import batch_size, epochs, lr, gamma, seed, seed_everything, build_model, device, train_dataset_path
from vit_prepare import get_train_valid_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

seed_everything(seed)
train_loader, valid_loader = get_train_valid_loaders(train_dataset_path)
model = build_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for data, label, path in tqdm(train_loader):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc/len(train_loader):.4f}")

    model.eval()
    val_acc = 0
    with torch.no_grad():
        for data, label, path in valid_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            acc = (output.argmax(dim=1) == label).float().mean()
            val_acc += acc.item()
    print(f"Validation Acc: {val_acc/len(valid_loader):.4f}")

torch.save(model.state_dict(), 'cats_dogs_vit.pth')
print("✅ 模型已保存为 cats_dogs_vit.pth")
