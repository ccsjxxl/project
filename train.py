import torch
import torch.nn as nn
import numpy as np
from src.utils import kl_divergence_loss
import torch.nn.functional as F
from datetime import datetime
import logging
import os
import psutil


def setup_logger():
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler("../result/50ing.log", mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


logger = setup_logger()


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cosine_temperature_annealing(epoch, max_epochs, initial_temp=1.0, T_min=0.1):
    """余弦退火温度调度器"""
    return T_min + 0.5 * (initial_temp - T_min) * (1 + np.cos(epoch / max_epochs * np.pi))


def log_memory_usage_epoch(logger, device):
    """每轮结束时的内存监控"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"[{get_timestamp()}] - [Epoch Memory] RSS={mem_info.rss / 1024 ** 2:.2f}MB | VMS={mem_info.vms / 1024 ** 2:.2f}MB")
    if device.type == 'cuda':
        logger.info(f"[{get_timestamp()}] - [Epoch GPU] Allocated={torch.cuda.memory_allocated(0) / 1024 ** 2:.2f}MB | Cached={torch.cuda.memory_reserved(0) / 1024 ** 2:.2f}MB")


def train_model(model, train_loader, val_loader, num_epochs, criterion,
                optimizer, scheduler, device, patience=5, initial_temp=1.0):
    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 动态温度调整（余弦退火）
        current_temp = cosine_temperature_annealing(epoch=epoch, max_epochs=num_epochs,
                                                    initial_temp=initial_temp, T_min=0.1)
        model.update_temperature(current_temp)

        # 动态KL系数调整（随epoch衰减）
        kl_weight = max(0.05, 0.2 * (0.95 ** epoch))

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, gate = model(x_batch)

            loss_ce = criterion(outputs, y_batch)
            loss_kl = kl_divergence_loss(gate)
            loss = loss_ce + kl_weight * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"[{get_timestamp()}]Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        val_loss = evaluate_model(model, val_loader, device, return_loss=True)
        logger.info(f"[{get_timestamp()}]Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"[{get_timestamp()}]Epoch {epoch + 1}, Learning Rate: {current_lr:.8f}")
        logger.info(f"[{get_timestamp()}]Current Temperature: {current_temp:.4f}, KL Weight: {kl_weight:.4f}")

        # 新增：每轮结束时输出内存使用情况
        log_memory_usage_epoch(logger, device)

        if counter >= patience:
            logger.info(f"[{get_timestamp()}]Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights:
        model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), "../result/50_final_model.pt")


def evaluate_model(model, test_loader, device, num_classes=None, return_loss=False, return_preds=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _ = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    avg_val_loss = total_loss / len(test_loader)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    if num_classes is None:
        num_classes = model.classifier[-1].out_features
    classes = list(range(num_classes))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    if num_classes > 2:
        all_labels_binarized = label_binarize(all_labels, classes=classes)
        try:
            auc = roc_auc_score(all_labels_binarized, all_probs,
                                average='weighted', multi_class='ovr')  # 改为ovr策略[4](@ref)
        except ValueError as e:
            if "Only one class present" in str(e):
                print(f"跳过AUC计算：验证集存在单类别问题，当前类别分布：{np.unique(all_labels)}")
                auc = 0.0  # 或设置为np.nan
            else:
                raise
    else:
        auc = roc_auc_score(all_labels, all_probs[:, 1])

    logger.info(f"[{get_timestamp()}]Metrics: Accuracy: {acc:.5f}, Precision: {prec:.5f}, Recall: {rec:.5f}, "
                f"F1 Score: {f1:.5f}, AUC-ROC: {auc:.5f}")

    if return_preds:
        return avg_val_loss if return_loss else (acc, prec, rec, f1, auc, all_labels, all_preds)
    else:
        return avg_val_loss if return_loss else (acc, prec, rec, f1, auc)

