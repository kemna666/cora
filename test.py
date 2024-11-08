import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset = Planetoid(root = './temp/cora',name = 'Cora',transform = NormalizeFeatures()) 
usingdevice = 'GPU' if device =='cuda:0' else 'cpu'
#确定一下是不是GPU跑
print(f'Using device: {usingdevice}')
#构建神经网络
class GCN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(GCN,self).__init__()
        self.Conv1 = GCNConv(input_dim,hidden_dim)
        self.Conv2 = GCNConv(hidden_dim,hidden_dim)
        # 全连接层
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self,x,edge_index):
        x = F.relu(self.Conv1(x, edge_index))
        x = F.relu(self.Conv2(x, edge_index))
        # 将图特征展平
        x = torch.flatten(x, start_dim=1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim = 1)

#准备数据
data = dataset[0].to(device)
input_dim = dataset.num_node_features
hidden_dim =10
output_dim = dataset.num_classes
lr = 0.01
EPOCH = 100

#模型和优化器，优化器用的Adam
model = GCN(input_dim,hidden_dim,output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr,weight_decay=5e-4)

#存储准确率list
loss_list = []
epoch_list =[]
accuracy_list = []
#训练模型
model.train()
for epoch in range(EPOCH):
    #梯度归零
    optimizer.zero_grad()
    out = model(data.x,data.edge_index)
    #负对数似然损失
    loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    epoch_list.append(epoch)
    loss_list.append(100-loss.item())


#使用matplotlib输出训练曲线图
plt.plot(epoch_list, loss_list, color='b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy_train')
plt.title('Training accuracy')
plt.show()

#测试模型
model.eval()
_,pred = model(data.x,data.edge_index).max(dim = 1)
correct = (pred[data.test_mask]==data.y[data.test_mask])
# 使用.sum()来计算正确预测的总数，并使用.item()获取标量值
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
total = data.test_mask.sum().item()
accuracy = correct / total
accuracy_list.append(accuracy)
print(f'accuracy = {accuracy:.4f}')
    
