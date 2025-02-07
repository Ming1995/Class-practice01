# 客製化BN層
# 將Gamma及Beta設置初始化為1及0使得最初的仿射變換成為一個恆等映射，及BN層僅執行標準化操作不會改變標準化後得輸出
class CustomBatchNorm(nn.Module): 
    def __init__(self,features,epsilon=1e-9):
        super(CustomBatchNorm,self).__init__()
        self.Gamma = nn.Parameter(torch.ones(features)) 
        self.Beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon # 防止除零產生錯誤
    def forward(self,x):
        mean = x.mean(dim=0,keepdim=True) # 沿著Batch維度計算均值
        var = x.var(dim=0,keepdim =True,unbiased=False) # 沿著Bathc維度計算變異數
        x_normalized = (x-mean)/torch.sqrt(var+self.epsilon)
        return self.Gamma.view(1,-1)*x_normalized+self.Beta.view(1,-1)
# 客製化tanh層
class Customtanh(nn.Module):
    def __init__(self):
        super(Customtanh,self).__init__()
        
    def forward(self,x):
        Output = (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
        return Output