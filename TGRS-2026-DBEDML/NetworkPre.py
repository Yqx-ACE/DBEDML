import torch
import torch.nn as nn
import torch.nn.functional as F
from AttnClassifier import Classifier
from BACKBONE import XYScanNet
import Energy_Loss
from DomainAlignment import DCADomainAligner

class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
    def forward(self, x):
        x = x.squeeze(0)
        x = self.trans_conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.mlp(x)
        return x

class FeatureNet(nn.Module):
    def __init__(self, way_num, embed_dim, feature_dim, params):
        super().__init__()
        self.embedding_dim = embed_dim
        self.feature_dim = feature_dim
        self.way_count = way_num
        self.feature_extractor = XYScanNet(params)
        self.classifier = Classifier(way_num, embed_dim)
        self.energy_loss = Energy_Loss.EnergyLoss(params)
        self.params = params
        self.domainAligner = DCADomainAligner(self.way_count + 1, params)
    def forward(self, inputs, targets=None, is_test=False):
        batch_sizes = [i.size(1) for i in inputs]
        batch, _, ch, h, w = inputs[0].shape
        x = torch.cat(inputs, dim=1).view(-1, ch, h, w)
        embed_feat, spatial_feat = self.feature_extractor(x)
        s_emb, q_emb, so_emb, o_emb, _ = torch.split(
            embed_feat.view(batch, -1, self.embedding_dim),
            batch_sizes, dim=1
        )
        s_spatial, q_spatial, so_spatial, o_spatial, domain_spatial = torch.split( 
            spatial_feat.view(batch, -1, self.embedding_dim, self.feature_dim, self.feature_dim),
            batch_sizes, dim=1
        )
        if targets is not None:
            s_tar, q_tar, so_tar, o_tar, _ = targets#,_
            tar = torch.cat([q_tar, o_tar], dim=1)
        s_emb = s_emb.view(batch, self.way_count, -1, self.embedding_dim)
        s_spatial = s_spatial.view(
            batch, self.way_count, -1, 
            self.embedding_dim, self.feature_dim, self.feature_dim
        )
        if is_test:
            emb_scores = self._calc_emb_protos((s_emb, q_emb, o_emb), tar, True)
            spa_scores = self._calc_spa_protos((s_spatial, q_spatial, o_spatial), tar, True)
            combined = ((1.0-self.params.fusion_weight) * emb_scores[0] + self.params.fusion_weight * spa_scores[0], (1.0-self.params.fusion_weight) * emb_scores[1] + self.params.fusion_weight * spa_scores[1])
            return self._gen_pred(*combined)[0]#, self.trans(q_spatial)
        emb_results = self._calc_emb_protos((s_emb, q_emb, o_emb), tar, False)
        spa_results = self._calc_spa_protos((s_spatial, q_spatial, o_spatial), tar, False)
        emb_scores, s_emb_proto, o_emb_proto, loss_cls_emb, loss_unit_emb = emb_results
        spa_scores, s_spa_proto, o_spa_proto, loss_cls_spa, loss_unit_spa = spa_results
        combined = ((1.0-self.params.fusion_weight) * emb_scores[0] + self.params.fusion_weight * spa_scores[0], (1.0-self.params.fusion_weight) * emb_scores[1] + self.params.fusion_weight * spa_scores[1])
        probs = self._gen_pred(*combined)
        loss_energy = self.energy_loss(emb_scores[0], emb_scores[1], spa_scores[0], spa_scores[1])
        
        so_emb = so_emb.view(batch, self.way_count, -1, self.embedding_dim)
        so_spatial = so_spatial.view(
            batch, self.way_count, -1, 
            self.embedding_dim, self.feature_dim, self.feature_dim
        )
        aug_emb = self._calc_emb_protos((so_emb, o_emb, q_emb), tar, False)
        aug_spa = self._calc_spa_protos((so_spatial, o_spatial, q_spatial), tar, False)
        emb_aug_scores, *_, loss_cls_aug_emb, loss_unit_aug_emb = aug_emb
        spa_aug_scores, *_, loss_cls_aug_spa, loss_unit_aug_spa = aug_spa
        loss_aug_energy = self.energy_loss(
            emb_aug_scores[0], emb_aug_scores[1],
            spa_aug_scores[0], spa_aug_scores[1]
        )
        
        loss_fsl = (
            (1.0-self.params.fusion_weight) * (loss_cls_emb + loss_cls_aug_emb) + self.params.fusion_weight * loss_energy + self.params.fusion_weight * (loss_cls_spa + loss_cls_aug_spa),
            (1.0-self.params.fusion_weight) * (loss_unit_emb + loss_unit_aug_emb) + self.params.fusion_weight * loss_aug_energy + self.params.fusion_weight * (loss_unit_spa + loss_unit_aug_spa)
        )
        
        '''-----------------------DomainAligner-----------------------'''
        q_flatten = q_spatial.view(batch, -1, self.embedding_dim*self.feature_dim*self.feature_dim)
        source_fea = torch.cat([s_spa_proto ,q_flatten, o_spa_proto], dim=1).unsqueeze(0)
        target_fea = domain_spatial.squeeze(0).view(-1,self.params.emd_dim * self.params.fea_dim * self.params.fea_dim).unsqueeze(0).unsqueeze(0)
        loss_align_s = self.domainAligner(source_fea, domain='source')
        loss_align_t = self.domainAligner(target_fea, domain='target')
        loss_da = loss_align_s['loss_adv_k'] + loss_align_t['loss_adv_k'] + loss_align_t['loss_adv_unk'] + loss_align_s['loss_adv_unk']
        loss = (loss_fsl,loss_da)
        return probs, loss
        

    def _calc_emb_protos(self, feats, tar, is_test):
        classifier_out = self.classifier(feats, 'emd')
        if is_test:
            return classifier_out[0]
        (q_scores, o_scores), s_proto, o_proto, unit_dist = classifier_out
        all_scores = torch.cat([q_scores, o_scores], dim=1)#torch.Size([1, 152, 9]) torch.Size([1, 152, 9])========>torch.Size([1, 304, 9])
        tar_flat = tar.view(-1)
        loss_cls = F.cross_entropy(
            all_scores.view(-1, self.way_count + 1), 
            tar_flat
        )
        loss_unit = fake_unit_compare(unit_dist, tar, self.way_count)
        return (q_scores, o_scores), s_proto, o_proto, loss_cls, loss_unit

    def _calc_spa_protos(self, feats, tar, is_test):
        classifier_out = self.classifier(feats, 'fea')
        if is_test:
            return classifier_out[0]
            
        (q_scores, o_scores), s_proto, o_proto, unit_dist = classifier_out
        all_scores = torch.cat([q_scores, o_scores], dim=1)
        tar_flat = tar.view(-1)
        loss_cls = F.cross_entropy(
            all_scores.view(-1, self.way_count + 1), 
            tar_flat
        )
        loss_unit = fake_unit_compare(unit_dist, tar, self.way_count)
        
        return (q_scores, o_scores), s_proto, o_proto, loss_cls, loss_unit

    def _gen_pred(self, q_scores, o_scores):
        q_probs = F.softmax(q_scores.detach(), dim=-1)
        o_probs = F.softmax(o_scores.detach(), dim=-1)
        return q_probs, o_probs

def fake_unit_compare(funit_distance, cls_label, n_ways):
    cls_label_binary = F.one_hot(cls_label, n_ways + 1)[:, :, :-1].float()#torch.Size([1, 304, 8])
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance,target=cls_label_binary))
    return loss