import pandas as pd

# 文件路径和标签列
# file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\UNSW_NB15_training-set.csv"
# file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\UNSW_NB15_testing-set.csv"
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Monday-WorkingHours.pcap_ISCX.csv"#1
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Tuesday-WorkingHours.pcap_ISCX.csv"#3
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Wednesday-workingHours.pcap_ISCX.csv"#6
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"#4
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"#2
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"#2
#file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"#2
file_path = r"D:\CNN_BiLSTM_Transformer\project\Data\Friday-WorkingHours-Morning.pcap_ISCX.csv"#2

#label_column = "label"  # UNSW_NB15数据集的类名
label_column = "Label"  # CIC-IDS2017数据集的类名

# 加载数据集，处理混合类型警告
data = pd.read_csv(file_path, low_memory=False)

# 检查列名
print("Columns in the dataset:", data.columns.tolist())

# 如果列名有多余空格，可清理列名
data.columns = data.columns.str.strip()

# 检查标签列是否存在
if label_column not in data.columns:
    raise KeyError(f"Label column '{label_column}' not found in the dataset.")

# 获取类别数量
num_classes = data[label_column].nunique()
print(f"Number of classes: {num_classes}")


