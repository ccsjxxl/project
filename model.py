import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFusion(nn.Module):
    def __init__(self, feature_dim, init_temp=1.0):
        super(DynamicFusion, self).__init__()
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        self.temperature = init_temp

    def update_temperature(self, new_temp):
        self.temperature = new_temp

    def forward(self, local_feat, global_feat):
        combined = torch.cat([local_feat, global_feat], dim=1)
        gate_logits = self.fc(combined) / self.temperature
        gate = torch.sigmoid(F.gelu(gate_logits))
        fused = gate * local_feat + (1 - gate) * global_feat
        return fused, gate

class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_heads=8, dropout=0.3, init_temp=1.0,
                 use_cnn=True, use_bilstm=False, use_transformer=True, use_dynamic_fusion=True):
        super(HybridModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_cnn = use_cnn
        self.use_bilstm = use_bilstm
        self.use_transformer = use_transformer
        self.use_dynamic_fusion = use_dynamic_fusion

        # CNN 模块
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

        # BiLSTM 模块
        if use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=hidden_dim if use_cnn else input_dim,
                hidden_size=hidden_dim,
                num_layers=4,
                bidirectional=True,
                batch_first=True,
                dropout=dropout
            )

        # Transformer 模块
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=2 * hidden_dim if use_bilstm else hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 动态融合模块
        if use_dynamic_fusion:
            self.fusion = DynamicFusion(feature_dim=2 * hidden_dim if use_bilstm or use_transformer else hidden_dim,
                                        init_temp=init_temp)

        # 分类器
        classifier_input_dim = 2 * hidden_dim if use_bilstm or use_transformer else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def update_temperature(self, new_temp):
        if self.use_dynamic_fusion:
            self.fusion.update_temperature(new_temp)

    def forward(self, x, temp=1.0):
        # x: [batch, seq_len, features]
        batch_size = x.shape[0]

        # CNN
        if self.use_cnn:
            x = x.permute(0, 2, 1)  # 调整维度以适配 CNN
            x = self.cnn(x)
            x = x.permute(0, 2, 1)  # 调整回 LSTM/Transformer 需要的形状

        # BiLSTM
        if self.use_bilstm:
            lstm_out, _ = self.bilstm(x)  # [batch, seq_len, 2 * hidden_dim]
        else:
            lstm_out = x  # 直接传递 CNN 结果或输入

        # Transformer
        if self.use_transformer:
            trans_out = self.transformer(lstm_out)  # [batch, seq_len, 2 * hidden_dim]
        else:
            trans_out = lstm_out  # 直接传递 LSTM 或 CNN 结果

        # 全局特征提取
        global_feat = torch.mean(trans_out, dim=1)  # [batch, hidden_dim or 2*hidden_dim]

        # 动态融合
        if self.use_dynamic_fusion:
            if self.use_cnn:
                cnn_global = torch.mean(x, dim=1)  # CNN 提取的全局特征
                cnn_global = F.pad(cnn_global, (0, self.hidden_dim)) if cnn_global.shape[1] != global_feat.shape[1] else cnn_global
            else:
                cnn_global = global_feat  # 没有 CNN 时，直接使用 Transformer/LSTM 结果
            fused_feat, gate = self.fusion(cnn_global, global_feat)
        else:
            fused_feat = global_feat  # 不使用动态融合时，直接使用 Transformer/LSTM 结果
            gate = None

        logits = self.classifier(fused_feat)
        return logits, gate
