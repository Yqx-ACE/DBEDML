import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyLoss(nn.Module):
    def __init__(self, args):
        super(EnergyLoss, self).__init__()
        self.args = args
        self.awl = AutomaticWeightedLoss(args, 2)
        self.m_in = torch.nn.Parameter(-torch.ones(1, requires_grad=True))
        self.m_out = torch.nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, f_klogits, f_ulogits, e_klogits, e_ulogits):
        args = self.args
        energy_loss_scale = 0.1
        if args.ahead_combine:
            k_energy_score = -torch.logsumexp(f_klogits, dim=1)
            u_energy_score = -torch.logsumexp(f_ulogits, dim=1)
            k_logits_score = -torch.logsumexp(e_klogits, dim=1)
            u_logits_score = -torch.logsumexp(e_ulogits, dim=1)
            k_energy_score = self.awl(k_energy_score,u_energy_score)
            u_energy_score = self.awl(k_logits_score,u_logits_score)

            if args.learnable_margin:
                l_energy = (torch.pow(F.relu(k_energy_score - self.m_in), 2).mean() + torch.pow(F.relu(self.m_out - u_energy_score), 2).mean()) * energy_loss_scale + torch.pow(F.relu(self.m_in - self.m_out), 2)
            else:
                l_energy = (torch.pow(F.relu(k_energy_score - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - u_energy_score), 2).mean()) * energy_loss_scale

        else:
            if args.energy_method == "sum":
                k_energy_score = -torch.logsumexp(f_klogits, dim=1)
                u_energy_score = -torch.logsumexp(f_ulogits, dim=1)
            else:
                k_energy_score = -torch.log(torch.max(torch.exp(f_klogits), dim=1)[0])
                u_energy_score = -torch.log(torch.max(torch.exp(f_ulogits), dim=1)[0])

            l_energy = (torch.pow(F.relu(k_energy_score - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - u_energy_score), 2).mean()) * energy_loss_scale

        return l_energy

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, args, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.args = args
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
    def forward(self, *x):
        loss_sum = 0
        if self.args.weighted_combine:
            for i, loss in enumerate(x):
                loss_sum += self.params[i] * loss
        else:
            for i, loss in enumerate(x):
                loss_sum += loss
        return loss_sum
