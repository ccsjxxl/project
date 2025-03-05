import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.data_loader import load_data, get_data_loaders
from src.model import HybridModel, NoCNNModel, NoLSTMModel, NoTransformerModel
from src.train import train_model, evaluate_model
from src.utils import plot_roc_curve, get_predictions_and_labels

# from sklearn.utils.class_weight import compute_class_weight

# 主函数
if __name__ == "__main__":

    # 数据集文件路径
    train_file = r"D:\CNN_BiLSTM_Transformer\project\Data\UNSW_NB15_training-set.csv"
    test_file = r"D:\CNN_BiLSTM_Transformer\project\Data\UNSW_NB15_testing-set.csv"
    label_column = "label"  # 标签列的名称（根据实际数据集确认）

    # 加载数据
    x_train, y_train, x_test, y_test, class_weights = load_data(train_file, test_file, label_column)  # 修改为解包5个返回值

    # 调试信息，确保行数匹配
    print(f"Length of x_train: {x_train.shape[0]}, Length of y_train: {y_train.shape[0]}")

    # 定义批量大小和数据加载器
    batch_size = 64
    train_loader, val_loader = get_data_loaders(x_train, y_train, x_test, y_test, batch_size)

    # # 数据集文件路径
    # train_file = r"D:\CNN_BiLSTM_Transformer\project\Data\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    #
    # test_file = r"D:\CNN_BiLSTM_Transformer\project\Data\Friday-WorkingHours-Morning.pcap_ISCX.csv"
    #
    # label_column = "Label"  # 确保标签列名称正确
    #
    #
    # # 加载数据
    # x_train, y_train, x_test, y_test, class_weights = load_data(train_file, test_file, label_column)
    #
    # # 调试信息，确保行数匹配
    # print(f"Length of x_train: {len(x_train)}, Length of y_train: {len(y_train)}")
    #
    # # 定义批量大小和数据加载器
    # batch_size = 64
    # train_loader, val_loader = get_data_loaders(x_train, y_train, x_test, y_test, batch_size)

    # 模型定义
    input_dim = x_train.shape[1]  # 特征维度（输入特征数量）
    # num_classes = len(np.unique(y_train))  # 类别数量
    # 获取实际类别数
    unique_labels = sorted(y_train.unique())  # 获取唯一标签
    num_classes = len(unique_labels)
    # model = HybridModel(input_dim=input_dim, hidden_dim=64, num_heads=8, num_classes=num_classes)
    # 使用原始 HybridModel
    model = HybridModel(input_dim=input_dim, hidden_dim=64, num_heads=8, num_classes=num_classes, dropout=0.3)

    # 使用消融实验模型
    # model = NoCNNModel(input_dim=input_dim, hidden_dim=64, num_heads=8, num_classes=num_classes, dropout=0.3)
    # model = NoLSTMModel(input_dim=input_dim, hidden_dim=64, num_heads=8, num_classes=num_classes, dropout=0.3)
    # model = NoTransformerModel(input_dim=input_dim, hidden_dim=64, num_heads=8, num_classes=num_classes, dropout=0.3)

    # 定义设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam 优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # 使用类别权重的交叉熵损失

    # 定义学习率调度器
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch学习率衰减为原来的0.1倍

    # 设置训练轮数
    num_epochs = 1  # 修改为实际需要的训练轮数

    # 训练模型
    # print("开始训练模型...")
    print(f"开始训练模型: {type(model).__name__}...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        patience=5,  # 早停容忍次数
        step_size=5,  # 学习率调度器步长
        gamma=0.1  # 学习率衰减因子
    )

    # 评估模型并输出最终指标
    print("评估模型性能...")
    metrics = evaluate_model(model, val_loader, device)
    print(f"模型评估指标: 准确率={metrics[0]:.4f}, 精确率={metrics[1]:.4f}, 召回率={metrics[2]:.4f}, "
          f"F1分数={metrics[3]:.4f}, AUC-ROC={metrics[4]:.4f}")

    # 提取预测结果和真实标签
    print("提取预测结果与真实标签...")
    y_true, y_pred = get_predictions_and_labels(model, val_loader, device)

    # 绘制 AUC-ROC 曲线
    print("绘制 AUC-ROC 曲线...")
    plot_roc_curve(y_true, y_pred, num_classes)

    print("主程序执行完毕。")
