from datetime import datetime
import torch
from torch import nn
from typing import Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

class HJBDataset():
    def __init__(self, domain_bsz, bound_bsz, xdim=250, T=1.0, rank=0):
        self.domain_bsz = domain_bsz
        self.bound_bsz = bound_bsz
        self.xdim = xdim
        self.T = T
        self.rank = rank

    def get_online_data(self):
        domain_X = torch.cat(
            [torch.randn((self.domain_bsz, self.xdim), device=self.rank), # x ~ N(0,1)
             torch.rand((self.domain_bsz, 1), device=self.rank)*self.T, # t ~ U(0,T)
            ],
            dim=1
        )
        
        boundary_X = torch.cat(
            [torch.randn((self.domain_bsz, self.xdim), device=self.rank), # x ~ N(0,1)
             torch.ones((self.domain_bsz, 1), device=self.rank)*self.T, # t = T
            ],
            dim=1
        )

        return domain_X, boundary_X

class HJBEquation:
    def __init__(self, x_dim, T, mu) -> None:
        self.xdim = x_dim
        self.mu = mu
        self.T = T
        self.sqrt_2 = 2**0.5

    def domain_loss(self, X, f, sample_cnt=None):
        dt = f.dt(X, sample_cnt)
        dx2 = f.dx2(X, sample_cnt)
        dx = f.dx(X, sample_cnt)
        residual = dt.squeeze(1) + torch.sum(dx2, dim=1) - self.mu*torch.sum(dx**2, dim=1)
        loss = torch.mean(residual**2)
        return loss

    def boundary_loss(self, X, f, sample_cnt=None):
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        x = X[:, :-1]
        gt = torch.log((1+ torch.sum(x**2, dim=1))/2)
        return torch.mean((y-gt)**2)

    def spatial_boundary_loss(self, X, f, sample_cnt=None):
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = self.ground_truth(X, sample_cnt)
        return torch.mean((y-gt)**2)

    def ground_truth(self, X, sample_cnt):
        batch_size = X.shape[0]
        x, t = X[:, :-1].unsqueeze(0), X[:, -1:].unsqueeze(0)
        sample_w = torch.normal(mean=0, std=1.0, size=(sample_cnt, batch_size, self.xdim), 
                                device=x.device, dtype=x.dtype)*torch.sqrt(self.T-t)
        sample_x = x + self.sqrt_2*sample_w
        sample_x2 = (1+torch.sum(sample_x**2, dim=2))/2
        g = torch.log(sample_x2)
        E = torch.mean(torch.exp(-self.mu*g), dim=0)
        u = -torch.log(E)/self.mu
        return u

class MLP(nn.Module):
    def __init__(self, layers:list):
        super(MLP, self).__init__()
        models = []
        for i in range(len(layers)-1):
            models.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                models.append(nn.Tanh())

        self.nn = nn.Sequential(*models)

    def forward(self, x):
        return self.nn(x)

class Wrapper:
    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class PinnWrapper(Wrapper):
    def __init__(self, g:nn.Module, x_dim) -> None:
        super().__init__()
        self.g:nn.Module = g
        self.x_dim = x_dim    

    def eval(self):
        self.g.eval()

    def train(self):
        self.g.train()

    def __call__(self, X, sample_cnt=None):
        with torch.set_grad_enabled(self.g.training):
            return self.g(X)
    
    def dx(self, X, sample_cnt=None):
        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)
        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        return df_dx

    def dt(self, X, sample_cnt=None):
        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)
        df_dx = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
        return df_dx

    def dx2(self, X, sample_cnt=None):
        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)

        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        d2f_dx2 = []
        for i in range(self.x_dim):
            # (batch_size, 1)
            d2f_dxidxi = torch.autograd.grad(df_dx[:, i].sum(), x, create_graph=True)[0][:, i:i+1]
            d2f_dx2.append(d2f_dxidxi)
        # (batch_size, x_dim)
        d2f_dx2 = torch.cat(d2f_dx2, dim=1)
        return d2f_dx2

def pgd(x, f, loss_func, step_cnt=5, step_size=0.2, t_lower_bound=0.0, t_upper_bound=1.0):
    for _ in range(step_cnt):
        x.requires_grad_()
        loss = loss_func(x, f)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x[:,-1] = torch.clamp(x[:,-1], t_lower_bound, t_upper_bound)
    return x

def build_wrapper(derivative, g):
    w_type = derivative
    if w_type == 'gt':
        wrapper = GroundTruthWrapper(mu, T, x_dim, sample_cnt)
    elif w_type == 'pinn':
        wrapper = PinnWrapper(g, x_dim)
    else:
        raise NotImplementedError
    return wrapper

def build_lr(model, lr, tot_epoch, warm_up_steps):
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr)
    warm_epoch = warm_up_steps
    lr_lambda = lambda epoch: 1-(epoch-warm_epoch)/(tot_epoch-warm_epoch) if epoch >= warm_epoch else epoch/warm_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda=lr_lambda)
    return optimizer_adam, scheduler

def test_l1(test_batch_size, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=test_batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = (pred_y - y).abs().sum()
            y_norm = y.abs().sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        avg_err = tot_err/X.shape[0]
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test_l2(test_batch_size, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=test_batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = ((pred_y - y)**2).sum()
            y_norm = (y**2).sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        tot_err, tot_norm = tot_err**0.5, tot_norm**0.5
        avg_err = tot_err/(X.shape[0]**0.5)
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test_w11(test_batch_size, f, X, Y, rank):
    f.eval()
    dataloader = DataLoader(TensorDataset(X, Y), batch_size=test_batch_size)
    tot_err, tot_norm = 0, 0
    for x, y in dataloader:
        x, y = x.to(rank), y.to(rank)
        pred_y = f(x)
        y_x = f.dx(x)
        pred_y = torch.cat([pred_y, y_x], dim=1)
        err = (pred_y - y).abs().sum()
        y_norm = y.abs().sum()
        tot_err += err.cpu().item()
        tot_norm += y_norm
    avg_err = tot_err/X.shape[0]
    rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test(cfg, f, X, Y, rank, norm_type='l1'):
    if norm_type == 'l1':
        return test_l1(cfg, f, X, Y, rank)
    elif norm_type == 'l2':
        return test_l2(cfg, f, X, Y, rank)
    elif norm_type == 'w11':
        return test_w11(cfg, f, X, Y, rank)
    else:
        raise NotImplementedError

# Constants for model
width = 4
depth = 4
sample_cnt = -1
derivative = "pinn"

# Constants for HJB Equation
mu = 1.0
T = 1.0
x_dim = 250
grad_step_cnt = 20
grad_step_size = 0.05

# Constants for HJB Dataset
domain_size = 50 # or 25
boundary_size = 50 # or 25
rank = "cpu"

# Constants for loss
domain_loss_constant = 1
boundary_loss_constant = 20
test_step = 5
test_batch_size = 1000

# Constants for test data path
test_data_path = "hjb_250_grad.pkl"

# Constants for training
iteration = 10000
lr = 7e-4
warm_up_steps = 0

layers = [x_dim + 1] + [width]*(depth - 1) + [1]
ddp_g = MLP(layers).to(rank)
f = build_wrapper(derivative, ddp_g)
dataset = HJBDataset(domain_bsz=domain_size, bound_bsz=boundary_size, xdim=x_dim, T=T, rank=rank)
hjb = HJBEquation(x_dim, T, mu)

with open(test_data_path, 'rb') as test_data_file:
    test_data = torch.load(test_data_file)
    test_X = test_data['X'].type(torch.FloatTensor)
    test_Y = test_data['Y'].type(torch.FloatTensor)
    test_grad_x = test_data['grad_x'].type(torch.FloatTensor)

optimizer, scheduler = build_lr(ddp_g, lr, iteration, warm_up_steps)

# These should go under training loop
for i in range(iteration):
    f.train()
    optimizer.zero_grad()

    domain_X, boundary_X = dataset.get_online_data()

    domain_X = pgd(domain_X, f, hjb.domain_loss, step_cnt=grad_step_cnt, step_size=grad_step_size, t_lower_bound=0, t_upper_bound=T)
    boundary_X = pgd(boundary_X, f, hjb.boundary_loss, step_cnt=grad_step_cnt, step_size=grad_step_size, t_lower_bound=T, t_upper_bound=T)

    dloss = hjb.domain_loss(domain_X, f)
    bloss = hjb.boundary_loss(boundary_X, f)
    loss = domain_loss_constant * dloss + boundary_loss_constant * bloss
    
    if (i+1) % test_step ==0:
        # test the model, only test in one thread.
        l1_i_avg_err, l1_i_rel_err = test(test_batch_size, f, test_X, test_Y, rank, norm_type='l1')
        l2_i_avg_err, l2_i_rel_err = test(test_batch_size, f, test_X, test_Y, rank, norm_type='l2')
        w11_i_avg_err, w11_i_rel_err = test(test_batch_size, f, test_X, torch.cat([test_Y[:, None], test_grad_x], dim=1), rank, norm_type='w11')
        print("-"*10 + "BEGIN LAMBDA: NONE ITERATION: " + str(i + 1))
        print(f'iteration {i}| loss {loss.detach().cpu().item():.5f}')
        print("L1 AVG. ERROR: " + str(l1_i_avg_err))
        print("L1 REL. ERROR: " + str(l1_i_rel_err))
        print("L2 AVG. ERROR: " + str(l2_i_avg_err))
        print("L2 REL. ERROR: " + str(l2_i_rel_err))
        print("W11 AVG. ERROR: " + str(w11_i_avg_err))
        print("W11 REL. ERROR: " + str(w11_i_rel_err))
    

    loss.backward()
    optimizer.step()
    scheduler.step()
