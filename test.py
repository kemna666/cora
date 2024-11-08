import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
dataset = Planetoid(root = './temp/cora',name = 'Cora',transform = NormalizeFeatures()) 

#构建神经网络
class GCN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(GCN,self).__init__()
        self.Conv1 = GCNConv(input_dim,hidden_dim)
        self.Conv2 = GCNConv(hidden_dim,output_dim)

    def forward(self,x,edge_index):
        x = self.Conv1(x,edge_index)
        x=self.Conv2(x,edge_index)
        x = F.relu(x)
        x = self.conv2(x,edge_index)
        return F.log_softmax(x,dim = 1)

#准备数据
data = dataset[0]
input_dim = dataset.num_node_features
hidden_dim = 16
output_dim = dataset.num_classes
lr = 0.01
EPOCH = 100

#模型和优化器，优化器用的Adam
model = GCN(input_dim,hidden_dim,output_dim)
optimizer = torch.optim.Adam(model.parameters(),lr,weight_decay=5e-4)

#训练模型
model.train()
for epoch in range(EPOCH):
    #梯度归零
    optimizer.zero_grad()
    out = model(data.x,data.edge_index)
    loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch %10 == 0 :
        print(f'{epoch},loss:{loss.item()}')


