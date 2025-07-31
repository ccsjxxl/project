import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import psutil  # 内存监控库
import os
from src.data_loader import load_data, get_data_loaders
from src.model import HybridModel
from src.train import train_model, evaluate_model
from src.utils import plot_roc_curve, get_predictions_and_labels
from datetime import datetime
from mixed_loss import MixedLoss
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def setup_logger():
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 文件 Handler，每次运行覆盖旧日志
    fh = logging.FileHandler("../result/50.log", mode="w")
    fh.setLevel(logging.INFO)
    # 日志格式设置
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 内存监控函数
def log_memory_usage(logger, device):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"{get_timestamp()} - 内存使用: RSS={mem_info.rss / 1024 ** 2:.2f}MB | VMS={mem_info.vms / 1024 ** 2:.2f}MB")
    if device.type == 'cuda':
        logger.info(f"{get_timestamp()} - GPU内存使用: Allocated={torch.cuda.memory_allocated(0) / 1024 ** 2:.2f}MB | Cached={torch.cuda.memory_reserved(0) / 1024 ** 2:.2f}MB")


# 绘制混淆矩阵函数
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # 在 PyCharm 中运行，不使用命令行参数，直接设置各模块状态
    # 默认情况下：CNN、BiLSTM、Transformer 以及动态融合均开启
    use_cnn = True
    use_bilstm = True
    use_transformer = True
    use_dynamic_fusion = True

    logger = setup_logger()
    logger.info("Starting main program execution with ablation settings:")
    logger.info(f"Use CNN: {use_cnn}, Use BiLSTM: {use_bilstm}, Use Transformer: {use_transformer}, Use Dynamic Fusion: {use_dynamic_fusion}")

    # 根据数据集选择参数（此处选择 UNSW-NB15 数据集）
    train_file = r"D:\CNN_BiLSTM_Transformer\project\Data\UNSW-NB15\UNSW_NB15_training-set.csv"
    test_file = r"D:\CNN_BiLSTM_Transformer\project\Data\UNSW-NB15\UNSW_NB15_testing-set.csv"
    label_column = "label"
    merge_data = False
    window_size = 128
    window_label_mode = "last"

    # 如果使用 CICIDS2017 数据集，可取消注释下面代码：
    # train_file = r"D:\CNN_BiLSTM_Transformer\project\Data\CICIDS2017\CICIDS2017_training-set2.csv"
    # test_file = r"D:\CNN_BiLSTM_Transformer\project\Data\CICIDS2017\CICIDS2017_testing-set2.csv"
    # label_column = "Label"
    # merge_data = True
    # window_size = None
    # window_label_mode = "last"

    # 加载数据
    x_train, y_train, x_test, y_test, class_weights = load_data(train_file, test_file, label_column, merge_data=merge_data)
    if merge_data:
        logger.info(f"After merging & splitting: Length of x_train: {x_train.shape[0]}, Length of y_train: {y_train.shape[0]}")
    else:
        logger.info(f"Length of x_train: {x_train.shape[0]}, Length of y_train: {y_train.shape[0]}")

    # 构造数据加载器
    batch_size = 64
    train_loader, val_loader = get_data_loaders(x_train, y_train, x_test, y_test,
                                                batch_size=batch_size,
                                                window_size=window_size,
                                                overlap=0.25,
                                                window_label_mode=window_label_mode)

    unique_labels = sorted(set(y_train) | set(y_test))
    num_classes = len(unique_labels)
    input_dim = x_train.shape[1]

    # 初始化模型，传入消融实验的参数
    model = HybridModel(input_dim=input_dim, hidden_dim=64, num_heads=8, num_classes=num_classes, dropout=0.3,
                        use_cnn=use_cnn, use_bilstm=use_bilstm, use_transformer=use_transformer, use_dynamic_fusion=use_dynamic_fusion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(f"{get_timestamp()} - 当前运行设备: {device.type.upper()}")
    if torch.cuda.is_available():
        logger.info(f"{get_timestamp()} - GPU型号: {torch.cuda.get_device_name(0)}")
    log_memory_usage(logger, device)

    # 使用混合损失
    criterion = MixedLoss(gamma=0.5, alpha=0.1, weight=class_weights.to(device), reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 50
    logger.info(f"{get_timestamp()} - 开始训练模型: {type(model).__name__} ...")
    train_model(model=model, train_loader=train_loader, val_loader=val_loader,
                num_epochs=num_epochs, criterion=criterion, optimizer=optimizer,
                scheduler=scheduler, device=device, patience=5, initial_temp=1.0)

    logger.info(f"{get_timestamp()} - 评估模型性能...")
    log_memory_usage(logger, device)
    metrics = evaluate_model(model, val_loader, device)
    logger.info(f"{get_timestamp()} - 模型评估指标: 准确率={metrics[0]:.5f}, 精确率={metrics[1]:.5f}, 召回率={metrics[2]:.5f}, F1分数={metrics[3]:.5f}, AUC-ROC={metrics[4]:.5f}")

    logger.info(f"{get_timestamp()} - 提取预测结果与真实标签...")
    log_memory_usage(logger, device)
    # 注意：get_predictions_and_labels 返回的是预测概率，需要转换为类别标签
    y_true, y_pred_probs = get_predictions_and_labels(model, val_loader, device)
    y_pred = np.argmax(y_pred_probs, axis=1)  # 转换为离散预测标签

    logger.info(f"{get_timestamp()} - 绘制 AUC-ROC 曲线...")
    log_memory_usage(logger, device)
    plot_roc_curve(y_true, y_pred_probs, num_classes)

    logger.info(f"{get_timestamp()} - 绘制混淆矩阵...")
    plot_confusion_matrix(y_true, y_pred, classes=[str(i) for i in range(num_classes)],
                          normalize=True, title='Normalized Confusion Matrix', save_path="../result/confusion_matrix2.png")

    logger.info(f"{get_timestamp()} - 主程序执行完毕。")
    log_memory_usage(logger, device)


