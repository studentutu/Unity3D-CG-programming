import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import time
from math import sqrt
from mesh_to_sdf import get_surface_point_cloud
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
import trimesh
import copy
import re

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, has_skip=False, skip_idx=1):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.has_skip = has_skip
        self.skip_idx = skip_idx
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)       
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1. / self.in_features, 1. / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        intermediate = torch.sin(self.omega_0 * self.linear(input))
        if self.has_skip:
            intermediate = intermediate/self.skip_idx + input
        return intermediate
   
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, omega=30, first_linear=False):
        super().__init__()
        self.omega = omega
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.first_linear=first_linear
        self.net = []
        if first_linear:
            linear = nn.Linear(in_features, hidden_features)
            with torch.no_grad():
                linear.weight.uniform_(-1. / self.in_features / omega, 1. / self.in_features / omega) 
            self.net.append(linear)
        else:
            self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega, has_skip=True, skip_idx=sqrt(i+1)))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega, np.sqrt(6 / hidden_features) / omega)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=omega))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

class SDFFitting(Dataset):
    def __init__(self, filename, samples):
        super().__init__()
        mesh = trimesh.load(filename)
        surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method='sample')
        self.coords, self.samples = surface_point_cloud.sample_sdf_near_surface(samples//2, use_scans=False, sign_method='normal')
        unit_sphere_points = sample_uniform_points_in_unit_sphere(samples//2)
        samples = surface_point_cloud.get_sdf_in_batches(unit_sphere_points, use_depth_buffer=False)
        self.coords = np.concatenate([self.coords, unit_sphere_points]).astype(np.float32)
        self.samples = np.concatenate([self.samples, samples]).astype(np.float32)      
        self.samples = torch.from_numpy(self.samples)[:,None]
        self.coords = torch.from_numpy(self.coords)
        print(self.coords.shape, self.samples.shape)
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError          
        return self.coords, self.samples

def train_network(dataloader, hidden_features, hidden_layers, omega):
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    img_curr = Siren(in_features=3, out_features=1, hidden_features=hidden_features, 
        hidden_layers=hidden_layers, outermost_linear=True, omega=omega, first_linear=False)
    img_curr.cuda()
    optim = torch.optim.Adam(lr=1e-4, params=img_curr.parameters(), weight_decay=.01)
    perm = torch.randperm(model_input.size(1))
    total_steps = 20000
    update = int(total_steps/50)
    batch_size = 256*256
    for step in range(total_steps):
        if step == 500: optim.param_groups[0]['weight_decay'] = 0.
        idx = step % int(model_input.size(1)/batch_size)
        model_in = model_input[:,perm[batch_size*idx:batch_size*(idx+1)],:]
        truth = ground_truth[:,perm[batch_size*idx:batch_size*(idx+1)],:]
        model_output, coords = img_curr(model_in)
        loss = (model_output - truth)**2
        loss = loss.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()          
        if (step % update) == update-1:
            perm = torch.randperm(model_input.size(1))
            print("Step %d, Current loss %0.6f" % (step, loss))
    return img_curr
    
def dump_data(dat):
    dat = dat.cpu().detach().numpy()
    return dat

def print_vec4(ws):
    vec = "vec4(" + ",".join(["{0:.3f}".format(w) for w in ws]) + ")"
    vec = re.sub(r"\b0\.", ".", vec)
    return vec

def print_mat4(ws):
    mat = "mat4(" + ",".join(["{0:.3f}".format(w) for w in np.transpose(ws).flatten()]) + ")"
    mat = re.sub(r"\b0\.", ".", mat)
    return mat        
    
def serialize_to_glsl(siren, varname):
    omega = siren.omega
    chunks = int(siren.hidden_features/4)
    lin = siren.net[0] if siren.first_linear else siren.net[0].linear
    in_w = dump_data(lin.weight)
    in_bias = dump_data(lin.bias)
    om = 1 if siren.first_linear else omega
    for row in range(chunks):
        if siren.first_linear:
            line = "vec4 %s0_%d=(" % (varname, row)
        else:
            line = "vec4 %s0_%d=sin(" % (varname, row)
        for ft in range(siren.in_features):
            feature = x_vec = in_w[row*4:(row+1)*4,ft]*om
            line += ("p.%s*" % ["y","z","x"][ft]) + print_vec4(feature) + "+"
        bias = in_bias[row*4:(row+1)*4]*om
        line += print_vec4(bias) + ");"
        print(line)
    for layer in range(siren.hidden_layers):
        layer_w = dump_data(siren.net[layer+1].linear.weight)
        layer_bias = dump_data(siren.net[layer+1].linear.bias)
        for row in range(chunks):
            line = ("vec4 %s%d_%d" % (varname, layer+1, row)) + "=sin("
            for col in range(chunks):
                mat = layer_w[row*4:(row+1)*4,col*4:(col+1)*4]*omega
                line += print_mat4(mat) + ("*%s%d_%d"%(varname, layer, col)) + "+\n    "
            bias = layer_bias[row*4:(row+1)*4]*omega
            line += print_vec4(bias)+")/%0.1f+%s%d_%d;"%(sqrt(layer+1), varname, layer, row)
            print(line)
    out_w = dump_data(siren.net[-1].weight)
    out_bias = dump_data(siren.net[-1].bias)
    for outf in range(siren.out_features):
        line = "return "
        for row in range(chunks):
            vec = out_w[outf,row*4:(row+1)*4]
            line += ("dot(%s%d_%d,"%(varname, siren.hidden_layers, row)) + print_vec4(vec) + ")+\n    "
        print(line + "{:0.3f}".format(out_bias[outf])+";")        
  
if __name__ == '__main__':
    sdf = SDFFitting("spot.obj", 256*256*4)
    sdfLoader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)
    neural_network = train_network(sdfLoader, 16, 2, 15)
    serialize_to_glsl(neural_network, "f")