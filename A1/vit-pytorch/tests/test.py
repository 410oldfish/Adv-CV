from vit_prepare import get_test_loader
from config import build_model, device, test_dataset_path
import torch

test_loader = get_test_loader(test_dataset_path)
model = build_model()
model.load_state_dict(torch.load('cats_dogs_vit.pth', map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, label, path in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        # 每张图片的预测结果和真实标签
        for i in range(len(pred)):
            true_label = 'dog' if label[i].item() == 1 else 'cat'
            pred_label = 'dog' if pred[i].item() == 1 else 'cat'
            print(f"✅ Predicted: {pred_label} | 🏷️ Actual: {true_label} | 📸 File: {path[i]}")

        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"🎯 Test Accuracy: {correct / total:.4f}")
