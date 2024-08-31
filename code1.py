from transformers import RoFormerModel,RoFormerTokenizer
import pandas as pd  
import torch
import torch.nn as nn 
import torch.optim as optim  
# 加载存档
model = RoFormerModel.from_pretrained("D:/code/MyNet/antiberta2-cssp")
tokenizer = RoFormerTokenizer.from_pretrained("D:/code/MyNet/antiberta2-cssp")

# 读取CSV文件  
df1 = pd.read_csv("model_2/data/testdata.csv")  
# df2 = pd.read_csv("model_2/data/bio_val_data.csv")  

# 获得train序列
train_Ab_sequences = []  
train_Ag_sequences = []  
for index, row in df1.iterrows(): 
    train_Ab_sequence = str(row['antibody_seq_a']) + '[SEP]'+ str(row['antibody_seq_b'])
    train_Ab_sequence = ' '.join(train_Ab_sequence).replace(' ', ' ').replace('[ S E P ]', '[SEP]')  
    train_Ag_sequence = str(row['antigen_seq']) 
    train_Ag_sequence = ' '.join(train_Ag_sequence).replace(' ', ' ').replace('[ S E P ]', '[SEP]')
    train_Ab_sequences.append(train_Ab_sequence)
    train_Ag_sequences.append(train_Ag_sequence)
# 获得val序列（测试时暂时先不用）
"""
val_Ab_sequences = []  
val_Ag_sequences = []  
for index, row in df2.iterrows(): 
    val_Ab_sequence = str(row['antibody_seq_a']) +'[SEP]' + str(row['antibody_seq_b'])
    val_Ab_sequence = ' '.join(val_Ab_sequence).replace(' ', ' ').replace('[ S E P ]', '[SEP]')  
    val_Ag_sequence = str(row['antigen_seq'])  
    val_Ag_sequence = ' '.join(val_Ag_sequence).replace(' ', ' ').replace('[ S E P ]', '[SEP]')
    val_Ab_sequences.append(val_Ab_sequence)
    val_Ag_sequences.append(val_Ag_sequence)
"""

#创建labels
train_labels = []
train_labels = df1['delta_g'].astype(float).dropna().tolist()
train_labels = torch.tensor(train_labels, dtype=torch.float)
train_labels = train_labels.unsqueeze(1) 
"""
val_labels = []
val_labels = df2['delta_g'].astype(float).dropna().tolist()
val_labels = torch.tensor(val_labels, dtype=torch.float)
val_labels = val_labels.unsqueeze(1) 
"""
# 设置读取时的窗口大小和步长
window_size = 256  # 必须小于或等于模型的最大位置嵌入
stride = 256
# 为什么到256就会开始报错，真的有这么短的抗原抗体序列吗，，，

# 使用预训练模型对训练集进行编码
train_list = []
train_Ab_list = []
for text in train_Ab_sequences:
    all_windows_outputs = []
    for i in range(0, len(text) - window_size + 1, stride):  
        window_text = text[i:i + window_size]  
        window_inputs = tokenizer(window_text, return_tensors="pt")  
        with torch.no_grad():  
            outputs = model(**window_inputs)  
        all_windows_outputs.append(outputs.last_hidden_state)  
    combined_output = torch.cat(all_windows_outputs, dim=1) 
    train_Ab_list.append(combined_output)

train_Ag_list = []
for text in train_Ag_sequences:
    all_windows_outputs = []
    for i in range(0, len(text) - window_size + 1, stride):  
        window_text = text[i:i + window_size]  
        window_inputs = tokenizer(window_text, return_tensors="pt")  
        with torch.no_grad():  
            outputs = model(**window_inputs)  
        all_windows_outputs.append(outputs.last_hidden_state)  
    combined_output = torch.cat(all_windows_outputs, dim=1) 
    train_Ag_list.append(combined_output)

#for i in range(len(train_Ab_list)):  
#    concatenated_tensor = torch.cat((train_Ab_list[i], train_Ag_list[i]), dim=1)  
#    train_list.append(concatenated_tensor)  

# 对验证集进行编码
"""
val_list = []
val_Ab_list = []
for text in val_Ab_sequences:
    all_windows_outputs = []
    inputs = tokenizer(text, return_tensors="pt")
    for i in range(0, inputs['input_ids'].size(1) - window_size + 1, stride):
    # 截取窗口
        window_inputs = {
            key: val[:, i:i + window_size] for key, val in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**window_inputs)
        all_windows_outputs.append(outputs.last_hidden_state)
    combined_output = torch.cat(all_windows_outputs, dim=1)
    val_Ab_list.append(combined_output)

val_Ag_list = []

for text in val_Ag_sequences:
    all_windows_outputs = []
    inputs = tokenizer(text, return_tensors="pt")
    for i in range(0, inputs['input_ids'].size(1) - window_size + 1, stride):
    # 截取窗口
        window_inputs = {
            key: val[:, i:i + window_size] for key, val in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**window_inputs)
        all_windows_outputs.append(outputs.last_hidden_state)
    combined_output = torch.cat(all_windows_outputs, dim=1)
    val_Ag_list.append(combined_output)
for i in range(len(val_Ab_list)):  
    concatenated_tensor = torch.cat((val_Ab_list[i], val_Ag_list[i]), dim=1)  
    val_list.append(concatenated_tensor)  
"""
TransformerEncoderLayer = nn.TransformerEncoderLayer
TransformerEncoder = nn.TransformerEncoder

class ScalarTransformer(torch.nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers):
        super(ScalarTransformer, self).__init__()
        encoder_layer1 = TransformerEncoderLayer(d_model, nhead)
        encoder_layer2 = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder1 = TransformerEncoder(encoder_layer1, num_encoder_layers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layer2, num_encoder_layers)
        self.linear1 = torch.nn.Linear(1024, 512)  
        self.linear2 = torch.nn.Linear(512, 128)  
        self.linear3 = torch.nn.Linear(128, 1)  
    def forward(self, Ab_list, Ag_list):
        out_list = []
        for x1,x2 in zip(Ab_list, Ag_list):
            src_mask = None  # 假设没有掩码
            x1 = self.transformer_encoder1(x1, src_mask)
            x2 = self.transformer_encoder2(x2, src_mask)
            combined = torch.cat((x1, x2), dim=1)  # 在最后一个维度上进行拼接
            processed = self.linear1(combined)
            processed = self.linear2(processed)
            processed = self.linear3(processed)
            processed = processed.mean(dim=1)
            out_list.append(processed) 
        return out_list
# 定义损失函数，例如使用交叉熵损失
criterion = torch.nn.MSELoss()
num_encoder_layers = 6 # 指定编码器层数
d_model = 1024
nhead = 8
model2 = ScalarTransformer(d_model=d_model, nhead=nhead, 
                                  num_encoder_layers=num_encoder_layers)
# 使用Adam优化器
optimizer = optim.Adam(model2.parameters(), lr=0.01)


# def train_model(model, criterion, optimizer, train_list, val_list, train_labels, val_labels, epochs):
def train_model(model2, criterion, optimizer, train_Ab_list, train_Ag_list, train_labels, epochs):
    for epoch in range(epochs):  
        model2.train(True) # 设置为训练模式
        optimizer.zero_grad() 
        output = model2(train_Ab_list,train_Ag_list) 
        output = torch.tensor(output, dtype=torch.float,requires_grad=True)
        loss = criterion(output, train_labels)  
        loss.backward()  # 反向传播  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # 梯度裁剪
        optimizer.step()  # 更新参数  
        if epoch % 10 == 0:
            model2.eval()  # 评估模式
            #with torch.no_grad():
            #    val_output = model(val_list)
            #    val_output = torch.tensor(val_output, dtype=torch.float,requires_grad=True)
            #    val_loss = criterion(val_output, val_labels)
            #    rmse = val_loss.item() ** (1/2)
            #print(f'Epoch {epoch+1}, Loss: {loss.item()}, Val RMSE: {rmse}') 
            print(f'Epoch {epoch+1}, Loss: {loss.item()}') 
#train_model(model, criterion, optimizer, train_list, val_list, train_labels, val_labels, epochs=100)
train_model(model2, criterion, optimizer, train_Ab_list, train_Ag_list, train_labels, epochs=100)