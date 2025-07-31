import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse


def load_data(train_file, test_file, label_column, merge_data=False, test_size=0.3, random_state=42):
    # 辅助函数：智能匹配列名（忽略大小写和空格）
    def find_column(df, col_name):
        col_name = str(col_name).strip().lower()
        for c in df.columns:
            if str(c).strip().lower() == col_name:
                return c
        available_cols = [str(col) for col in df.columns]
        raise ValueError(
            f"Column '{col_name}' not found in DataFrame. "
            f"Available columns: {available_cols}"
        )

    # 加载数据
    if merge_data:
        if isinstance(train_file, list):
            train_data = pd.concat([pd.read_csv(f) for f in train_file], ignore_index=True)
        else:
            train_data = pd.read_csv(train_file)
        if isinstance(test_file, list):
            test_data = pd.concat([pd.read_csv(f) for f in test_file], ignore_index=True)
        else:
            test_data = pd.read_csv(test_file)
        data = pd.concat([train_data, test_data], ignore_index=True)
        data.columns = data.columns.str.strip()
        actual_label = find_column(data, label_column)
    else:
        if isinstance(train_file, list):
            train_data = pd.concat([pd.read_csv(f) for f in train_file], ignore_index=True)
        else:
            train_data = pd.read_csv(train_file)
        if isinstance(test_file, list):
            test_data = pd.concat([pd.read_csv(f) for f in test_file], ignore_index=True)
        else:
            test_data = pd.read_csv(test_file)

        train_data.columns = train_data.columns.str.strip()
        test_data.columns = test_data.columns.str.strip()
        actual_label = find_column(train_data, label_column)
        try:
            find_column(test_data, label_column)
        except ValueError as e:
            raise ValueError(f"Test data missing label column: {e}")

    # 数据清洗
    if merge_data:
        data[actual_label] = data[actual_label].astype(str).fillna('')
        data[actual_label] = data[actual_label].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)
    else:
        train_data[actual_label] = train_data[actual_label].astype(str).fillna('')
        test_data[actual_label] = test_data[actual_label].astype(str).fillna('')
        train_data[actual_label] = train_data[actual_label].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)
        test_data[actual_label] = test_data[actual_label].str.strip().str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # 分割特征和标签
    if merge_data:
        X = data.drop(columns=[actual_label])
        y = data[actual_label]
    else:
        X = train_data.drop(columns=[actual_label])
        y = train_data[actual_label]
        X_test = test_data.drop(columns=[actual_label])
        y_test = test_data[actual_label]

    # 处理缺失值和无穷值
    def clean_data(X_data):
        X_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_data.dropna(inplace=True)
        return X_data

    X = clean_data(X)
    if not merge_data:
        X_test = clean_data(X_test)

    # 对齐数据长度
    if merge_data:
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
    else:
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        min_len = min(len(X_test), len(y_test))
        X_test = X_test.iloc[:min_len]
        y_test = y_test.iloc[:min_len]

    # 特征预处理
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X = pipeline.fit_transform(X)
    if not merge_data:
        X_test = pipeline.transform(X_test)

    # 数据集划分
    if merge_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        combined_labels = y_train
    else:
        X_train, y_train, X_test, y_test = X, y, X_test, y_test
        combined_labels = y

    # 标签编码
    unique_labels = sorted(combined_labels.unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    if merge_data:
        y_train = y_train.map(label_mapping)
        y_test = y_test.map(label_mapping)
    else:
        y_train = y_train.map(label_mapping)
        y_test = y_test.map(label_mapping)

    print("Label mapping:", label_mapping)
    print("Mapped y_train unique values:", np.unique(y_train))
    print("Mapped y_test unique values:", np.unique(y_test))

    # 计算类别权重
    full_classes = np.array(range(len(label_mapping)))
    y_train_np = np.array(y_train)
    counts = np.array([np.sum(y_train_np == i) for i in full_classes])
    total = len(y_train_np)
    weights = np.array([total / (len(full_classes) * c) if c > 0 else 1.0 for c in counts])
    class_weights = torch.tensor(weights, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, class_weights


def create_sliding_windows(x, y, window_size, overlap=0.25, label_mode='last'):
    """滑动窗口生成函数（新增CIC-IDS2017专用模式）"""
    if issparse(x):
        x = x.toarray()

    # 新增CIC-IDS2017专用逻辑 [3,4](@ref)
    if label_mode == 'attack_presence':
        # 边界填充防止攻击事件切割
        pad_size = window_size // 2
        x_padded = np.pad(x, ((pad_size, pad_size), (0, 0)), mode='edge')
        y_padded = np.pad(np.array(y), (pad_size, pad_size), mode='edge')

        # 计算滑动步长
        step = max(1, int(window_size * (1 - overlap)))
        windows = []
        window_labels = []

        # 生成窗口和标签
        for i in range(0, len(x_padded) - window_size + 1, step):
            window = x_padded[i:i + window_size]
            window_y = y_padded[i:i + window_size]

            # 存在即标记逻辑：窗口内有攻击则标记为1，否则0
            label = 0 if (window_y == 0).all() else 1
            windows.append(window)
            window_labels.append(label)

        return np.array(windows, dtype=np.float32), np.array(window_labels)

    # 原有逻辑保持不变
    step = max(1, int(window_size * (1 - overlap)))
    windows = []
    window_labels = []
    max_index = x.shape[0] - window_size + 1

    for i in range(0, max_index, step):
        window = x[i:i + window_size]
        windows.append(window)

        if label_mode == 'last':
            window_labels.append(y.iloc[i + window_size - 1] if hasattr(y, 'iloc') else y[i + window_size - 1])
        elif label_mode == 'majority':
            from scipy.stats import mode
            window_labels.append(mode(y.iloc[i:i + window_size] if hasattr(y, 'iloc') else y[i:i + window_size])[0][0])
        elif label_mode == 'any':
            window_vals = y.iloc[i:i + window_size] if hasattr(y, 'iloc') else y[i:i + window_size]
            counts = pd.Series(window_vals).value_counts()
            if 0 in counts:
                benign_count = counts[0]
            else:
                benign_count = 0
            if benign_count < window_size:
                counts_non_benign = counts.drop(0, errors='ignore')
                if not counts_non_benign.empty:
                    window_labels.append(counts_non_benign.idxmax())
                else:
                    window_labels.append(0)
            else:
                window_labels.append(0)
        else:
            raise ValueError("Unknown label_mode: choose 'last', 'majority', 'any' or 'attack_presence'")

    return np.array(windows, dtype=np.float32), np.array(window_labels)


def get_data_loaders(x_train, y_train, x_test, y_test, batch_size=64, window_size=None, overlap=0.25,
                     window_label_mode='last'):
    # 滑动窗口处理
    if window_size is not None:
        x_train, y_train = create_sliding_windows(x_train, y_train, window_size, overlap, label_mode=window_label_mode)
        x_test, y_test = create_sliding_windows(x_test, y_test, window_size, overlap, label_mode=window_label_mode)
        print(f"\nAfter sliding window processing:")
        print(f"Train features: {x_train.shape}, labels: {len(y_train)}")
        print(f"Test features: {x_test.shape}, labels: {len(y_test)}")
    else:
        seq_len = 10
        x_train = np.repeat(np.expand_dims(x_train, axis=1), repeats=seq_len, axis=1)
        x_test = np.repeat(np.expand_dims(x_test, axis=1), repeats=seq_len, axis=1)
        print(f"\nNo sliding window processing, data shapes:")
        print(f"Train features: {x_train.shape}, labels: {len(y_train)}")
        print(f"Test features: {x_test.shape}, labels: {len(y_test)}")

    # 转换为numpy数组
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 转换为PyTorch张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
