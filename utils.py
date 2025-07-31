import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import logging

# 配置 logger，如果在调用处已有全局 logger，此处也可直接使用
logger = logging.getLogger("UtilsLogger")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_predictions_and_labels(model, data_loader, device, label_mapping=None):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            y_pred.append(probs.cpu())
            y_true.append(labels.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if label_mapping:
        # 将数值标签转换为原始标签（字符串），仅用于显示参考
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        y_true = [reverse_mapping[label] for label in y_true]
    return y_true, y_pred

def plot_roc_curve(y_true, y_pred, num_classes, label_names=None):
    """
    绘制 ROC 曲线。如果提供 label_names 参数，则它应为一个字典，
    键为类别索引，值为对应的原始标签。这样图例中就会显示原始标签而不是 "Class i"。
    注意：y_true 需要为数值标签（例如 0,1,2...），如果在 get_predictions_and_labels 中转换为字符串，
    则这里需要根据 label_names 将其重新映射为对应的数值比较。
    """
    # 如果 y_true 是字符串形式，则将其转换回数字形式（假设 label_names 中的值与原始标签一致）
    if y_true.dtype.type is np.str_ or y_true.dtype.type is np.object_:
        # 假设 label_names 字典中 key 是数值，value 是原始标签
        reverse_map = {v: k for k, v in label_names.items()}
        y_true_numeric = np.array([reverse_map[label] for label in y_true])
    else:
        y_true_numeric = np.array(y_true)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        # 将 y_true_numeric 转换为二值标签：1 表示该样本属于类别 i，0 否则
        binary_true = (y_true_numeric == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_true, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        logger.info(f"Class {i}: AUC = {roc_auc[i]:.5f}")
    plt.figure()
    for i in range(num_classes):
        if label_names is not None:
            # 使用 label_names 字典转换索引到原始标签
            label_str = label_names.get(i, f"Class {i}")
        else:
            label_str = f"Class {i}"
        plt.plot(fpr[i], tpr[i], label=f"{label_str} AUC: {roc_auc[i]:.5f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("AUC-ROC Curve")
    plt.show()


def kl_divergence_loss(gate, target_value=0.5):
    # 如果 gate 为 None，则返回零损失
    if gate is None:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    eps=1e-8
    kl = gate * torch.log((gate + eps) / (target_value + eps)) + \
         (1 - gate) * torch.log(((1 - gate) + eps) / ((1 - target_value) + eps))
    return torch.mean(kl)
