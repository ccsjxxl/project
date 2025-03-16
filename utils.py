# import matplotlib.pyplot as plt
# import torch
# from sklearn.metrics import roc_curve, auc
#
#
# def get_predictions_and_labels(model, data_loader, device, label_mapping=None):
#     model.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             probs = torch.softmax(outputs, dim=1)  # 转换为概率
#             y_pred.append(probs.cpu())
#             y_true.append(labels.cpu())
#     y_true = torch.cat(y_true).numpy()
#     y_pred = torch.cat(y_pred).numpy()
#
#     # 可选：解码真实标签为文本
#     if label_mapping:
#         reverse_mapping = {v: k for k, v in label_mapping.items()}
#         y_true = [reverse_mapping[label] for label in y_true]
#
#     return y_true, y_pred  # 返回真实标签（可选解码）和预测概率
#
#
# def plot_roc_curve(y_true, y_pred, num_classes):
#     """
#     绘制 AUC-ROC 曲线。
#     :param y_true: 真实标签
#     :param y_pred: 模型预测概率
#     :param num_classes: 类别数量
#     """
#     fpr, tpr, roc_auc = {}, {}, {}
#     for i in range(num_classes):
#         # 使用多类标签的概率进行绘制
#         fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     plt.figure()
#     for i in range(num_classes):
#         plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC: {roc_auc[i]:.2f}")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend(loc="best")
#     plt.title("AUC-ROC Curve")
#     plt.show()
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc

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
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        y_true = [reverse_mapping[label] for label in y_true]

    return y_true, y_pred

def plot_roc_curve(y_true, y_pred, num_classes):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC: {roc_auc[i]:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("AUC-ROC Curve")
    plt.show()

def kl_divergence_loss(gate, target_value=0.5):
    """
    计算门控权重与目标分布（均匀分布，目标均值为 0.5）之间的 KL 散度正则化损失，
    用于鼓励门控输出不过早饱和。
    """
    eps = 1e-8
    kl = gate * torch.log((gate + eps) / (target_value + eps)) + \
         (1 - gate) * torch.log(((1 - gate) + eps) / ((1 - target_value) + eps))
    return torch.mean(kl)
