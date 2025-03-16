# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.preprocessing import label_binarize
# import numpy as np
#
#
# def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, patience, step_size,
#                 gamma):
#     model.to(device)
#
#     # 获取所有标签以计算类别权重
#     all_labels = []
#     for _, y_batch in train_loader:
#         all_labels.extend(y_batch.cpu().numpy())  # 确保从 GPU 转移到 CPU
#
#     # 在此处输出唯一标签
#     print("Unique labels in y_train:", torch.unique(torch.tensor(all_labels)))  # 这行代码打印唯一标签
#
#     class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
#     print(f"Class Weights: {class_weights}")  # 确保权重正确应用
#     class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
#
#     # 使用加权的交叉熵损失
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#     # 定义学习率调度器
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#
#     best_val_loss = float('inf')  # 初始化最佳验证损失
#     counter = 0  # 早停计数器
#
#     for epoch in range(num_epochs):
#         model.train()  # 训练模式
#         train_loss = 0.0
#
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#
#             optimizer.zero_grad()  # 清空梯度
#             outputs = model(x_batch)  # 前向传播
#             loss = criterion(outputs, y_batch)  # 计算损失
#             loss.backward()  # 反向传播
#             optimizer.step()  # 参数更新
#
#             train_loss += loss.item()
#
#         avg_train_loss = train_loss / len(train_loader)  # 计算平均训练损失
#         print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
#
#         # 验证集评估
#         val_loss = evaluate_model(model, val_loader, device, return_loss=True)
#         print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
#
#         # 更新学习率
#         scheduler.step()
#         current_lr = scheduler.get_last_lr()[0]
#         print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.6f}")
#
#         # 早停逻辑
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             counter = 0
#         else:
#             counter += 1
#
#         if counter >= patience:
#             print(f"Early stopping at epoch {epoch + 1}")
#             break
#
#
# def evaluate_model(model, test_loader, device, return_loss=False):
#     """
#     模型评估，支持返回评估指标和验证损失。
#
#     参数:
#     - model: 已训练的模型
#     - test_loader: 测试集 DataLoader
#     - device: 计算设备（CPU 或 GPU）
#     - return_loss: 是否返回验证损失（默认 False）
#
#     返回:
#     - 如果 return_loss=True，返回验证损失
#     - 如果 return_loss=False，返回评估指标（准确率、精确率、召回率、F1 分数、AUC-ROC）
#     """
#     model.eval()  # 评估模式
#     criterion = nn.CrossEntropyLoss()  # 验证集不考虑权重
#     all_labels = []
#     all_preds = []
#     total_loss = 0.0
#
#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             total_loss += loss.item()
#
#             preds = torch.argmax(outputs, dim=1)  # 预测结果
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y_batch.cpu().numpy())
#
#     # 计算验证损失
#     avg_val_loss = total_loss / len(test_loader)
#
#     # 计算评估指标
#     unique_labels = sorted(set(all_labels))
#     all_labels_binarized = label_binarize(all_labels, classes=unique_labels)
#     all_preds_binarized = label_binarize(all_preds, classes=unique_labels)
#     acc = accuracy_score(all_labels, all_preds)
#     prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#     rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
#     auc = roc_auc_score(all_labels_binarized, all_preds_binarized, average='weighted', multi_class='ovr')
#
#     print(f"Metrics: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, "
#           f"AUC-ROC: {auc:.4f}")
#
#     return avg_val_loss if return_loss else (acc, prec, rec, f1, auc)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from src.utils import kl_divergence_loss

def temperature_annealing(initial_temp, epoch, k=0.01, T_min=0.5):
    """
    指数退火策略：随着训练轮次增加，温度逐步降低，
    使得动态门控模块更平滑地过渡到稳定状态。
    """
    return max(T_min, initial_temp * np.exp(-k * epoch))

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, patience, initial_temp):
    model.to(device)

    # 计算类别权重（仅用于打印和定义损失函数）
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    print("Unique labels in y_train:", torch.unique(torch.tensor(all_labels)))
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    print(f"Class Weights: {class_weights}")
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 使用加权交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 更新温度参数
        new_temp = temperature_annealing(initial_temp, epoch)
        model.update_temperature(new_temp)
        print(f"Epoch {epoch+1}, Temperature: {new_temp:.4f}")

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, gate = model(x_batch)
            loss_ce = criterion(outputs, y_batch)
            loss_kl = kl_divergence_loss(gate)
            loss = loss_ce + 0.1 * loss_kl  # 0.1 为 KL 损失系数，可调
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(model, val_loader, device, return_loss=True)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def evaluate_model(model, test_loader, device, return_loss=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _ = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_val_loss = total_loss / len(test_loader)
    unique_labels = sorted(set(all_labels))
    all_labels_binarized = label_binarize(all_labels, classes=unique_labels)
    all_preds_binarized = label_binarize(all_preds, classes=unique_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    auc = roc_auc_score(all_labels_binarized, all_preds_binarized, average='weighted', multi_class='ovr')

    print(f"Metrics: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc:.4f}")
    return avg_val_loss if return_loss else (acc, prec, rec, f1, auc)
