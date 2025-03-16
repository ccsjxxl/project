# import numpy as np
# import pandas as pd
# # from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import compute_class_weight
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split  # 导入分层抽样
# from imblearn.over_sampling import SMOTE  # 导入 SMOTE
# import torch
# from torch.utils.data import DataLoader, TensorDataset
#
#
# def load_data(train_file, test_file, label_column):
#     """
#     加载并处理数据，包括重新映射标签，使标签值从0开始连续。
#     :param train_file: 训练集文件路径（str 或 list[str]）
#     :param test_file: 测试集文件路径（str 或 list[str]）
#     :param label_column: 标签列名称
#     :return: 预处理后的训练集和测试集（特征、标签、类别权重）
#     """
#     # 加载单文件或合并多文件的训练数据
#     if isinstance(train_file, list):
#         train_data = pd.concat([pd.read_csv(file) for file in train_file], ignore_index=True)
#     else:
#         train_data = pd.read_csv(train_file)
#
#     if isinstance(test_file, list):
#         test_data = pd.concat([pd.read_csv(file) for file in test_file], ignore_index=True)
#     else:
#         test_data = pd.read_csv(test_file)
#
#     # 清理列名，移除多余空格
#     train_data.columns = train_data.columns.str.strip()
#     test_data.columns = test_data.columns.str.strip()
#
#     # 确保标签列为字符串类型
#     train_data[label_column] = train_data[label_column].astype(str).fillna('')
#     test_data[label_column] = test_data[label_column].astype(str).fillna('')
#
#     # 去除非ASCII字符
#     train_data[label_column] = train_data[label_column].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)
#     test_data[label_column] = test_data[label_column].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)
#
#     # 分离特征和标签
#     x_train = train_data.drop(columns=[label_column])
#     y_train = train_data[label_column]
#     x_test = test_data.drop(columns=[label_column])
#     y_test = test_data[label_column]
#
#     # 合并训练和测试标签以进行统一映射
#     combined_labels = pd.concat([y_train, y_test], axis=0)
#     unique_labels = sorted(combined_labels.unique())  # 确保标签按顺序排列
#     label_mapping = {label: idx for idx, label in enumerate(unique_labels)}  # 创建映射字典
#
#     # 应用标签映射
#     y_train = y_train.map(label_mapping)
#     y_test = y_test.map(label_mapping)
#
#     # # 使用分层抽样来划分训练集和测试集
#     # x_train, x_test, y_train, y_test = train_test_split(
#     #     x_train, y_train, test_size=0.2, stratify=y_train  # 使用 stratify 确保类别分布一致
#     # )
#
#     # 检查结果
#     print("Label mapping:", label_mapping)
#     print("Mapped y_train unique values:", y_train.unique())
#     print("Mapped y_test unique values:", y_test.unique())
#
#     # 数据清洗：移除无穷值和 NaN
#     x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
#     x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
#     x_train.dropna(inplace=True)
#     x_test.dropna(inplace=True)
#
#     # 确保 x_train 和 y_train 的长度一致
#     min_len = min(len(x_train), len(y_train))
#     x_train = x_train.iloc[:min_len]
#     y_train = y_train.iloc[:min_len]
#
#     min_len = min(len(x_test), len(y_test))
#     x_test = x_test.iloc[:min_len]
#     y_test = y_test.iloc[:min_len]
#
#     # 检查特征类型
#     num_features = x_train.select_dtypes(include=["int64", "float64"]).columns
#     cat_features = x_train.select_dtypes(include=["object"]).columns
#
#     # 定义数值和类别特征的处理器
#     num_transformer = StandardScaler()  # 对数值特征进行标准化
#     cat_transformer = OneHotEncoder(handle_unknown="ignore")  # 对类别特征进行独热编码
#
#     # 构建列变换器
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", num_transformer, num_features),
#             ("cat", cat_transformer, cat_features),
#         ]
#     )
#
#     # 使用管道处理数据
#     pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
#     x_train = pipeline.fit_transform(x_train)
#     x_test = pipeline.transform(x_test)
#
#     # # 使用 SMOTE 平衡训练数据
#     # smote = SMOTE(random_state=42)
#     # x_train, y_train = smote.fit_resample(x_train, y_train)
#
#     # 计算类别权重
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     class_weights = torch.tensor(class_weights, dtype=torch.float32)
#
#     # 将不连续标签重新映射为连续整数
#     unique_labels = sorted(set(y_train) | set(y_test))  # 确保训练和测试一致
#     label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
#     y_train = y_train.map(label_mapping)
#     y_test = y_test.map(label_mapping)
#
#     return x_train, y_train, x_test, y_test, class_weights
#
#
# def create_sliding_windows(x, window_size, overlap=0.25):
#     """
#     对时间序列数据进行滑动窗口分块。
#     假设 x 的形状为 (num_timesteps, feature_dim)，返回形状为 (num_windows, window_size, feature_dim)。
#     """
#     step = int(window_size * (1 - overlap))
#     windows = []
#     for i in range(0, x.shape[0] - window_size + 1, step):
#         windows.append(x[i:i+window_size])
#     return np.array(windows, dtype=np.float32)
#
# def get_data_loaders(x_train, y_train, x_test, y_test, batch_size=64,window_size=None, overlap=0.25):
#     """
#     创建 DataLoader，用于批量加载数据。
#     :param x_train: 训练集特征
#     :param y_train: 训练集标签
#     :param x_test: 测试集特征
#     :param y_test: 测试集标签
#     :param batch_size: 批量大小
#     :return: 训练集和测试集的 DataLoader
#     """
#     # 滑动窗口
#     if window_size is not None:
#         x_train = create_sliding_windows(x_train, window_size, overlap)
#         y_train = y_train[window_size - 1:]
#         x_test = create_sliding_windows(x_test, window_size, overlap)
#         y_test = y_test[window_size - 1:]
#
#     # 确保稀疏矩阵被转换为密集数组
#     if hasattr(x_train, 'toarray'):
#         x_train = x_train.toarray()
#     if hasattr(x_test, 'toarray'):
#         x_test = x_test.toarray()
#
#     # 将数据转换为 PyTorch 张量
#     x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
#     x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
#
#     # 确保标签被转换为长整型张量
#     y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
#     y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
#
#     # 创建训练和测试数据的 TensorDataset
#     train_tensor = TensorDataset(x_train_tensor, y_train_tensor)
#     test_tensor = TensorDataset(x_test_tensor, y_test_tensor)
#
#     # 创建训练和测试的 DataLoader
#     train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
#
#     return train_loader, test_loader

import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse  # 导入稀疏矩阵检查工具


def load_data(train_file, test_file, label_column):
    """
    加载并处理数据，包括重新映射标签，使标签值从0开始连续。
    :param train_file: 训练集文件路径（str 或 list[str]）
    :param test_file: 测试集文件路径（str 或 list[str]）
    :param label_column: 标签列名称
    :return: 预处理后的训练集和测试集（特征、标签、类别权重）
    """
    # 加载单文件或合并多文件的训练数据
    if isinstance(train_file, list):
        train_data = pd.concat([pd.read_csv(file) for file in train_file], ignore_index=True)
    else:
        train_data = pd.read_csv(train_file)

    if isinstance(test_file, list):
        test_data = pd.concat([pd.read_csv(file) for file in test_file], ignore_index=True)
    else:
        test_data = pd.read_csv(test_file)

    # 清理列名，移除多余空格
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    # 确保标签列为字符串类型
    train_data[label_column] = train_data[label_column].astype(str).fillna('')
    test_data[label_column] = test_data[label_column].astype(str).fillna('')

    # 去除非ASCII字符
    train_data[label_column] = train_data[label_column].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)
    test_data[label_column] = test_data[label_column].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # 分离特征和标签
    x_train = train_data.drop(columns=[label_column])
    y_train = train_data[label_column]
    x_test = test_data.drop(columns=[label_column])
    y_test = test_data[label_column]

    # 合并训练和测试标签以进行统一映射
    combined_labels = pd.concat([y_train, y_test], axis=0)
    unique_labels = sorted(combined_labels.unique())  # 确保标签按顺序排列
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}  # 创建映射字典

    # 应用标签映射
    y_train = y_train.map(label_mapping)
    y_test = y_test.map(label_mapping)

    # 数据清洗：移除无穷值和 NaN
    x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_train.dropna(inplace=True)
    x_test.dropna(inplace=True)

    # 确保 x_train 和 y_train 的长度一致
    min_len = min(len(x_train), len(y_train))
    x_train = x_train.iloc[:min_len]
    y_train = y_train.iloc[:min_len]

    min_len = min(len(x_test), len(y_test))
    x_test = x_test.iloc[:min_len]
    y_test = y_test.iloc[:min_len]

    # 检查特征类型
    num_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    cat_features = x_train.select_dtypes(include=["object"]).columns

    # 定义数值和类别特征的处理器
    num_transformer = StandardScaler()  # 对数值特征进行标准化
    cat_transformer = OneHotEncoder(handle_unknown="ignore")  # 对类别特征进行独热编码

    # 构建列变换器
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    # 使用管道处理数据
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    x_train = pipeline.fit_transform(x_train)
    x_test = pipeline.transform(x_test)

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # 将不连续标签重新映射为连续整数
    unique_labels = sorted(set(y_train) | set(y_test))  # 确保训练和测试一致
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = y_train.map(label_mapping)
    y_test = y_test.map(label_mapping)

    return x_train, y_train, x_test, y_test, class_weights


def create_sliding_windows(x, window_size, overlap=0.25):
    """
    对时间序列数据进行滑动窗口分块。
    假设 x 的形状为 (num_timesteps, feature_dim)，返回形状为 (num_windows, window_size, feature_dim)。
    """
    # 如果输入是稀疏矩阵，转换为密集矩阵
    if issparse(x):
        x = x.toarray()

    step = int(window_size * (1 - overlap))
    windows = []
    for i in range(0, x.shape[0] - window_size + 1, step):
        windows.append(x[i:i+window_size])
    return np.array(windows, dtype=np.float32)


def get_data_loaders(x_train, y_train, x_test, y_test, batch_size=64, window_size=None, overlap=0.25):
    """
    创建 DataLoader，用于批量加载数据。
    :param x_train: 训练集特征
    :param y_train: 训练集标签
    :param x_test: 测试集特征
    :param y_test: 测试集标签
    :param batch_size: 批量大小
    :return: 训练集和测试集的 DataLoader
    """
    # 滑动窗口
    if window_size is not None:
        x_train = create_sliding_windows(x_train, window_size, overlap)
        y_train = y_train[window_size - 1:]
        x_test = create_sliding_windows(x_test, window_size, overlap)
        y_test = y_test[window_size - 1:]

    # 确保稀疏矩阵被转换为密集数组
    if hasattr(x_train, 'toarray'):
        x_train = x_train.toarray()
    if hasattr(x_test, 'toarray'):
        x_test = x_test.toarray()

    # 将数据转换为 PyTorch 张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # 确保标签被转换为长整型张量
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # 创建训练和测试数据的 TensorDataset
    train_tensor = TensorDataset(x_train_tensor, y_train_tensor)
    test_tensor = TensorDataset(x_test_tensor, y_test_tensor)

    # 创建训练和测试的 DataLoader
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
