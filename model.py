import torch.nn as nn
import torch.nn.functional as F
import torch

def stratified_norm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    # out_str = out.clone()
    for i in range(n_subs):
        out[n_samples*i: n_samples*(i+1), :] = (out[n_samples*i: n_samples*(i+1), :] - out[n_samples*i: n_samples*(i+1), :].mean(
            dim=0)) / (out[n_samples*i: n_samples*(i+1), :].std(dim=0) + 1e-3)
    return out

def batch_norm(out):
    # out_str = out.clone()
    out_str = (out - out.mean(dim=0)) / (out.std(dim=0) + 1e-3)
    return out_str

def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        # out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(
            0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str

def batch_layerNorm(out):
    n_samples, chn1, chn2, n_points = out.shape
    out = out.reshape(n_samples, -1, n_points).permute(0,2,1)
    out = out.reshape(n_samples*n_points, -1)
    out_str = (out - out.mean(dim=0)) / (out.std(dim=0) + 1e-3)
    out_str = out_str.reshape(n_samples, n_points, chn1*chn2).permute(
        0,2,1).reshape(n_samples, chn1, chn2, n_points)
    return out_str

class ConvNet_baseNonlinearHead(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact, isMaxPool, args):
        super(ConvNet_baseNonlinearHead, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 30))
        # self.bn1 = nn.BatchNorm2d(n_timeFilters)
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 6), groups=n_timeFilters*multiFact)
        # self.bn2 = nn.BatchNorm2d(n_timeFilters*multiFact*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
        self.isMaxPool = isMaxPool
        self.args = args
        self.classfier = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )

    def forward(self, input):
        # print(input.shape)
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))

        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        out = F.elu(out)
        out = self.avgpool(out)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))
        # elif 'middle1_batch' in self.stratified:
        #     out = self.bn1(out)

        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))
        # print(out.shape)

        if 'middle2' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))
        # elif 'middle2_batch' in self.stratified:
        #     # out = batch_layerNorm(out)
        #     out = self.bn2(out)

        if self.isMaxPool:
            # Select the dim with max average values (half of the total dims)
            _, indices = torch.topk(out.mean(dim=3), out.shape[1]//2, dim=1)
            out_pooled = torch.zeros((out.shape[0], out.shape[1]//2, out.shape[2], out.shape[3])).to(self.args.device)
            for i in range(out.shape[0]):
                out_pooled[i,:,:,:] = out[i,indices[i,:,0]]
            out_pooled = out_pooled.reshape(out_pooled.shape[0], -1)
            return out_pooled, indices
        else:
            out = out.reshape(out.shape[0], -1)
            cl_out = self.classfier(out)
            return out, cl_out


class ConvNet_baseNonlinearHead_learnRescale(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, multiFact):
        super(ConvNet_baseNonlinearHead_learnRescale, self).__init__()
        self.rescaleConv1 = nn.Conv2d(1, 1, (1, timeFilterLen))
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.rescaleConv2 = nn.Conv2d(1, 1, (1, timeFilterLen))
        self.avgpool = nn.AvgPool2d((1, 30))
        # self.bn1 = nn.BatchNorm2d(n_timeFilters)
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 6), groups=n_timeFilters*multiFact)
        self.rescaleConv3 = nn.Conv2d(1, 1, (1, 6))
        # self.bn2 = nn.BatchNorm2d(n_timeFilters*multiFact*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
    def forward(self, input):
        out_tmp = self.rescaleConv1(input)
        out_mean = torch.mean(out_tmp, 3, True)
        out_var = torch.mean(out_tmp**2, 3, True)
        input = (input - out_mean) / torch.sqrt(out_var + 1e-5)

        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)

        out = out.reshape(out.shape[0], 1, out.shape[1]*out.shape[2], out.shape[3])
        out_tmp = self.rescaleConv2(out)
        out_mean = torch.mean(out_tmp, 3, True)
        out_var = torch.mean(out_tmp**2, 3, True)
        out = (out - out_mean) / torch.sqrt(out_var + 1e-5)
        out = out.reshape(out.shape[0], self.n_timeFilters, self.n_spatialFilters, out.shape[3])

        out = F.elu(out)
        out = self.avgpool(out)

        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))

        out = out.permute(0,2,1,3)
        out_tmp = self.rescaleConv3(out)
        out_mean = torch.mean(out_tmp, 3, True)
        out_var = torch.mean(out_tmp**2, 3, True)
        out = (out - out_mean) / torch.sqrt(out_var + 1e-5)
        out = out.permute(0,2,1,3)

        out = out.reshape(out.shape[0], -1)
        return out




class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, n_samples, stratified):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_samples = n_samples
        self.stratified = stratified
    def forward(self, input):
        if self.stratified:
            input = stratified_norm(input, self.n_samples)
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        if self.stratified:
            out = stratified_norm(out, self.n_samples)
        out = self.fc3(out)
        return out

class LSTM_NN(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, n_samples, stratified):
        super(LSTM_NN, self).__init__()
        self.lstm = nn.LSTM(inp_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_samples = n_samples
        self.stratified = stratified
    def forward(self, input):
        # input: (batch, seq, features)
        input, _ = self.lstm(input)
        input = input.reshape(input.shape[0]*input.shape[1], -1)
        if self.stratified:
            input = stratified_norm(input, self.n_samples)
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        if self.stratified:
            out = stratified_norm(out, self.n_samples)
        out = self.fc3(out)
        return out