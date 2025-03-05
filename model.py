import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes, dropout=0.3):
        """
        混合模型，用于处理时间序列数据。
        模型结合了 CNN、双向 LSTM 和 Transformer，并最终通过全连接层实现分类。

        参数:
        - input_dim (int): 输入特征的维度。
        - hidden_dim (int): LSTM 和 Transformer 的隐藏层维度。
        - num_heads (int): Transformer 的多头注意力头数。
        - num_classes (int): 分类任务中的类别数量。
        - dropout (float): Dropout 概率，用于防止过拟合。
        """
        super(HybridModel, self).__init__()

        # CNN 模块: 提取局部时序特征
        self.cnn = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # 双向 LSTM 模块: 提取长时依赖关系
        self.bilstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=2, bidirectional=True, batch_first=True, dropout=dropout
        )

        # Transformer 模块: 建模全局上下文
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2, nhead=num_heads,
                dropout=dropout, batch_first=True
            ),
            num_layers=2
        )

        # Dropout 层: 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 全连接层模块: 实现最终分类
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 层归一化模块: 稳定特征分布
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        """
        定义模型的前向传播流程。

        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, sequence_length, feature_dim]。

        返回:
        - out (Tensor): 模型预测的类别分布，形状为 [batch_size, num_classes]。
        """
        # 增加通道维度以适配 CNN 模块
        x = x.unsqueeze(1)  # [batch_size, 1, sequence_length, feature_dim]
        x = self.cnn(x).squeeze(1)  # [batch_size, sequence_length, feature_dim]

        # 通过 BiLSTM 提取时间序列特征
        lstm_out, _ = self.bilstm(x)  # [batch_size, sequence_length, 2 * hidden_dim]

        # 对 LSTM 输出进行归一化处理
        lstm_out = self.layer_norm(lstm_out)

        # 通过 Transformer 进一步建模
        transformer_out = self.transformer(lstm_out)  # [batch_size, sequence_length, 2 * hidden_dim]

        # 平均池化提取全局特征
        pooled = transformer_out.mean(dim=1)  # [batch_size, 2 * hidden_dim]

        # Dropout 防止过拟合
        pooled = self.dropout(pooled)

        # 全连接层实现分类
        out = self.fc(pooled)  # [batch_size, num_classes]

        return out



class NoCNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes, dropout):
        """
        初始化 NoCNNModel
        参数：
        - input_dim: 输入特征维度
        - hidden_dim: LSTM 隐藏层维度
        - num_heads: 保留参数，但未使用
        - num_classes: 分类任务类别数
        - dropout: Dropout 概率
        """
        super(NoCNNModel, self).__init__()

        # 双向 LSTM 模块
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # 全连接分类层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        前向传播
        参数：
        - x: [batch_size, sequence_length, feature_dim]
        返回：
        - out: [batch_size, num_classes]
        """
        # 确保输入有时间维度
        if x.dim() == 2:  # 如果输入缺少时间维度
            x = x.unsqueeze(1)  # 添加时间维度 -> [batch_size, 1, feature_dim]

        # LSTM 输出: [batch_size, sequence_length, hidden_dim * 2]
        lstm_out, _ = self.bilstm(x)

        # 平均池化以获得 [batch_size, hidden_dim * 2]
        pooled = lstm_out.mean(dim=1)

        # 全连接层输出: [batch_size, num_classes]
        out = self.fc(pooled)

        return out


class NoLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes, dropout=0.3):
        super(NoLSTMModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2, nhead=num_heads,
                dropout=dropout, batch_first=True
            ),
            num_layers=2
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # # 检查输入形状
        # print(f"Input shape: {x.shape}")

        # 如果缺少序列维度，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加序列维度 -> [batch_size, 1, input_dim]

        # 确保输入是 [batch_size, sequence_length, input_dim]
        if x.size(1) < x.size(2):
            x = x.transpose(1, 2)  # 转置为 [batch_size, input_dim, sequence_length]

        x = self.cnn(x)  # CNN 提取特征: [batch_size, hidden_dim * 2, sequence_length]
        x = x.transpose(1, 2)  # 转置回 Transformer 需要的格式: [batch_size, sequence_length, hidden_dim * 2]
        transformer_out = self.transformer(x)  # Transformer 编码
        pooled = transformer_out.mean(dim=1)  # 平均池化
        pooled = self.dropout(pooled)
        out = self.fc(pooled)  # 全连接层
        return out


class NoTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes, dropout=0.3):
        super(NoTransformerModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim * 2, hidden_size=hidden_dim,
            num_layers=2, bidirectional=True, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        # print(f"Input shape before unsqueeze: {x.shape}")
        x = x.unsqueeze(1)
        # print(f"Shape after unsqueeze: {x.shape}")
        x = self.cnn(x).squeeze(1)
        # print(f"Shape after CNN: {x.shape}")
        x = x.transpose(1, 2)  # 转换为 [batch_size, sequence_length, feature_dim]
        lstm_out, _ = self.bilstm(x)
        # print(f"Shape after BiLSTM: {lstm_out.shape}")
        lstm_out = self.layer_norm(lstm_out)
        # print(f"Shape after LayerNorm: {lstm_out.shape}")
        pooled = lstm_out.mean(dim=1)
        # print(f"Shape after pooling: {pooled.shape}")
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        # print(f"Shape after fully connected layer: {out.shape}")
        return out



