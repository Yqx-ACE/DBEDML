import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None
revgrad = RevGrad.apply

class RevGrad(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = torch.tensor(alpha, requires_grad=False)
    def forward(self, input_):
        return revgrad(input_, self._alpha)
def grad_reverse(x, lambd=1.0):
    return RevGrad(lambd)(x)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_class = args.known_class + 1
        self.generator = ResBase()
        self.classifier = Classifier(self.num_class, unit_size=1024)
 
        dim = 1024 if args.all_layer_adv else 2048
        self.adv_k = AdversarialNetwork(dim)
        self.adv_unk = AdversarialNetwork(dim)

    def forward(self, x, constant=1, adaption=False):
        rois = self.generator(x)
        x = self.classifier(rois, constant, adaption)
        return  x

class Classifier(nn.Module):
    def __init__(self, num_classes, feature_dim, unit_size=32):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(feature_dim, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.classifier = nn.Linear(unit_size, num_classes)
        self.drop = nn.Dropout(p=0.3)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, rois, constant=1, adaption=False, pooling=True, return_feat=False):
        if pooling:
            rois = self.average_pooling(rois).view(rois.size(0), -1)
        x = self.drop(F.relu(self.bn1(self.linear1(rois))))
        x = self.drop(F.relu(self.bn2(self.linear2(x))))
        x_rev = grad_reverse(x, constant) if adaption else x
        logits = self.classifier(x_rev)
        if return_feat:
            return logits, x
        else:
            return logits

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.main1 = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True,),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant=0.05):
        x = grad_reverse(x, constant)
        for module in self.main1.children():
            x = module(x)
        return x.view(-1)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def enable_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

class DCADomainAligner(nn.Module):
    def __init__(self, class_nums, args, adv_grl = 0.1):
        super().__init__()
        self.class_nums = class_nums
        self.feature_dim = args.emd_dim
        self.classifier = Classifier(self.class_nums, self.feature_dim)
        self.criterion_bce_red = nn.BCELoss(reduction='none')
        self.adv_grl = adv_grl
        self.fix_bn = fix_bn
        self.enable_bn = enable_bn
        self.adv_k = AdversarialNetwork(self.feature_dim)
        self.adv_unk = AdversarialNetwork(self.feature_dim)
        self.GDG = nn.Linear(args.fea_dim * args.fea_dim * args.emd_dim, self.feature_dim)
    def forward(self, rois, all_layers=False, domain='source'):
        self.classifier.apply(self.fix_bn)
        self.GDG.apply(self.fix_bn)
        domain_label = 1.0 if domain == 'source' else 0.0
        bs, c, h, w = rois.size()
        
        
        rois_flatten = rois.contiguous().view(-1, w)
        rois_flatten = self.GDG(rois_flatten)
        with torch.set_grad_enabled(all_layers):
            if not all_layers:
                scores = self.classifier(rois_flatten, pooling=False).softmax(-1).detach()
            else:
                scores, rois_flatten = self.classifier(rois_flatten, pooling=False, return_feat=True)
                scores = scores.softmax(-1).detach()
        self.classifier.apply(self.enable_bn)
        target = torch.full((rois_flatten.size(0),),domain_label,dtype=torch.float,device=rois_flatten.device)
        
        
        
        
        weight_unk = scores[:, -1]
        weight_k = scores[:, :-1].sum(-1)
        adv_k = self.adv_k(rois_flatten, self.adv_grl)
        adv_unk = self.adv_unk(rois_flatten, self.adv_grl)
        loss_adv_k = (self.criterion_bce_red(adv_k, target) * weight_k).mean()
        loss_adv_unk = (self.criterion_bce_red(adv_unk, target) * weight_unk).mean()
        
        
        
        return {
            'loss_adv_k': loss_adv_k,
            'loss_adv_unk': loss_adv_unk
        }