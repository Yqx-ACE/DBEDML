import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb

class ExpertNetwork(nn.Module):
    def __init__(self, in_dim, n_head=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.n_head = n_head
        self.d_k = in_dim // n_head
        assert in_dim % n_head == 0, "in_dim must be divisible by n_head"
        self.w_q = nn.Linear(in_dim, in_dim)
        self.w_k = nn.Linear(in_dim, in_dim)
        self.w_v = nn.Linear(in_dim, in_dim)
        self.fc = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.semantic_proj = nn.Linear(300, in_dim) if 300 else None
    def forward(self, x, res=True):
        x = x.unsqueeze(1)
        k = x
        v = x
        residual = x
        batch_size, seq_len, _ = x.size()
        q = self.w_q(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.in_dim)
        output = self.fc(context)
        output = self.dropout(output)
        if res:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)
        return output.squeeze(1)
class SparseMoEGate(nn.Module):
    def __init__(self, in_dim, num_experts=8, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(in_dim, num_experts)
    def forward(self, x):
        logits = self.gate(x)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=1)
        gate_weights = torch.zeros_like(logits)
        for i in range(self.top_k):
            gate_weights.scatter_(1, top_indices[:, [i]], F.softmax(top_values[:, i], dim=0).unsqueeze(1))        
        return gate_weights, top_indices
class SparseMoEFakePrototypeGenerator(nn.Module):
    def __init__(self, d_model, num_experts=8, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = d_model[0]
        self.output_dim = d_model[0]
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertNetwork(in_dim=self.input_dim)
            for _ in range(num_experts)
        ])
        self.gate = SparseMoEGate(in_dim=self.input_dim, num_experts=num_experts)
        self.semantic_proj = nn.Linear(300, self.input_dim) if 300 else None
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(self.input_dim)
    def forward(self, q,k,v, res=True):
        batch_size, seq_len, input_dim = q.size()
        residual = q
        q_flat = q.reshape(-1, self.input_dim)
        gate_weights, _ = self.gate(q_flat)
        expert_outputs = torch.zeros_like(q_flat)
        for expert_id in range(self.num_experts):
            expert_mask = gate_weights[:,expert_id].unsqueeze(1)
            expert_input = q_flat * expert_mask
            expert_out = self.experts[expert_id](expert_input)
            expert_outputs += expert_out * expert_mask
        output = expert_outputs.reshape(batch_size, seq_len, self.output_dim)
        output = self.dropout(output)
        if res:
            output = self.layernorm(output + residual)
        else:
            output = self.layernorm(output)        
        return output

class Classifier(nn.Module):
    def __init__(self, n_ways: int, feat_dim: int):
        super().__init__()
        self.n_ways = n_ways
        self.feat_dim = feat_dim
        self.calibrator = SupportCalibrator(n_ways, feat_dim, 1)
        self.open_gen = OpenSetGenerater(feat_dim, 1)
        self.metric = Metric_Cosine()
        self.metric_fea = get_feat_logits(n_ways, feat_dim)
        self.metric_emb = EuclideanDistanceCalculator()
    def _get_protos(self, s_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_agg = torch.mean(s_feat, dim=2)
        supp_proto = self.calibrator(s_agg)
        fake_proto, recip_unit = self.open_gen(supp_proto, self.n_ways)
        return supp_proto, fake_proto, recip_unit
    def forward(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mode: str) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        s_feat, q_feat, o_feat = feats
        supp_proto, fake_proto, recip_unit = self._get_protos(s_feat)
        all_proto = torch.cat([supp_proto, fake_proto], dim=1)
        if mode == 'emd':
            q_score = self.metric(all_proto, q_feat)
            o_score = self.metric(all_proto, o_feat)
            q_dist = 1.0 + self.metric(recip_unit, q_feat)
            o_dist = 1.0 + self.metric(recip_unit, o_feat)#相似度越高，这个值就趋近于2
            cls_score = (q_score,o_score)
            unit_dist = (q_dist,o_dist)
            #print(q_dist.shape)
            return cls_score, supp_proto, fake_proto, torch.cat([unit_dist[0], unit_dist[1]], dim=1)
            
        if mode == 'fea':
            h = s_feat.shape[-1]
            all_proto = all_proto.view(1, self.n_ways+1, self.feat_dim, h, h)
            recip_unit = recip_unit.view(1, self.n_ways, self.feat_dim, h, h)
            cls_score = self.metric_fea(all_proto, q_feat, o_feat, "pixel_sim")
            unit_dist = self.metric_fea(recip_unit, q_feat, o_feat)#越相似，这个值就越趋近于0，负数后+1就越趋近于1
            return cls_score, supp_proto, fake_proto, torch.cat([1 - unit_dist[0], 1 - unit_dist[1]], dim=1)
            
            
            
class SupportCalibrator(nn.Module):
    def __init__(self, nway: int, feat_dim: int, n_head: int = 1):
        super().__init__()
        self.nway = nway
        self.feat_dim = feat_dim
        self.attn = SparseMoEFakePrototypeGenerator(d_model=[feat_dim,feat_dim], num_experts=4, hidden_dim=128, dropout=0.1)
    def forward(self, s_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_feat = s_feat.view(-1, 1, self.feat_dim)
        s_center = self.attn(flat_feat, flat_feat, flat_feat)
        return s_center.view(s_feat.shape[0], self.nway, -1)#, s_attn.view(s_feat.shape[0], self.nway, -1)
class OpenSetGenerater(nn.Module):
    def __init__(self, feat_dim: int, n_head: int = 1):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_basic = SparseMoEFakePrototypeGenerator(d_model=[feat_dim,feat_dim], num_experts=4, hidden_dim=128, dropout=0.1)
        self.agg_basic = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, feat_dim)
        )
        self.attn_1600 = SparseMoEFakePrototypeGenerator(d_model=[1600,1600], num_experts=4, hidden_dim=128, dropout=0.1)
        self.agg_1600 = nn.Sequential(
            nn.Linear(1600, 1600),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(1600, 1600)
        )
    def forward(self, s_center: torch.Tensor, n_ways: int) -> tuple[torch.Tensor, torch.Tensor]:
        bs = s_center.shape[0]
        if s_center.shape[-1] == self.feat_dim:
            flat_feat = s_center.view(n_ways, 1, self.feat_dim)
            func_unit = self.attn_basic(flat_feat, flat_feat, flat_feat)
            func_unit = func_unit.view(bs, -1, self.feat_dim)
            fake_proto = self.agg_basic(func_unit.mean(dim=1, keepdim=True))
            return fake_proto, func_unit
        flat_feat = s_center.view(n_ways, 1, -1)
        func_unit = self.attn_1600(flat_feat, flat_feat, flat_feat)
        func_unit = func_unit.view(bs, -1, 1600)
        fake_proto = self.agg_1600(func_unit.mean(dim=1, keepdim=True))
        return fake_proto, func_unit

class get_feat_logits(nn.Module):
    def __init__(self, num_ways: int, feat_dim: int, temperature: float = 10):
        super().__init__()
        self.num_ways = num_ways
        self.feat_dim = feat_dim
        self.temp = temperature
        self.top_k = 5
        self.strategy = 'mix'
    def _euclidean(self, proto: torch.Tensor, kq: torch.Tensor, uq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_flat = proto.view(proto.size(0), proto.size(1), -1)
        kq_flat = kq.view(kq.size(0), kq.size(1), -1)
        uq_flat = uq.view(uq.size(0), uq.size(1), -1)
        p_broad = p_flat.unsqueeze(1)  # [B, 1, N, D]
        kq_broad = kq_flat.unsqueeze(2)  # [B, M, 1, D]
        uq_broad = uq_flat.unsqueeze(2)  # [B, U, 1, D]
        k_logits = torch.sum((p_broad - kq_broad) ** 2, dim=-1) / self.temp
        u_logits = torch.sum((p_broad - uq_broad) ** 2, dim=-1) / self.temp
        return k_logits, u_logits
    def _pixel_similarity(self, proto: torch.Tensor, kq: torch.Tensor, uq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_flat = proto.squeeze().view(-1, proto.size(2), proto.size(-1) * proto.size(-2)).unsqueeze(0).unsqueeze(2)
        kq_flat = kq.view(-1, kq.size(2), kq.size(-1)*kq.size(-2)).unsqueeze(1).permute(0, 1, 3, 2).unsqueeze(-1)
        uq_flat = uq.view(-1, uq.size(2), uq.size(-1)*uq.size(-2)).unsqueeze(1).permute(0, 1, 3, 2).unsqueeze(-1)
        k_sim = torch.nn.CosineSimilarity(dim=3)(kq_flat, p_flat)
        u_sim = torch.nn.CosineSimilarity(dim=3)(uq_flat, p_flat)
        if self.strategy == 'query':
            k_logits = k_sim.topk(self.top_k, dim=3).values.sum(dim=[2, 3]) / self.top_k
            u_logits = u_sim.topk(self.top_k, dim=3).values.sum(dim=[2, 3]) / self.top_k
        elif self.strategy == 'proto':
            k_logits = k_sim.topk(self.top_k, dim=2).values.sum(dim=[2, 3]) / self.top_k
            u_logits = u_sim.topk(self.top_k, dim=2).values.sum(dim=[2, 3]) / self.top_k
        else:
            k_top = k_sim.topk(self.top_k, dim=2).values / self.top_k
            u_top = u_sim.topk(self.top_k, dim=2).values / self.top_k
            k_logits = k_top.topk(self.top_k, dim=3).values.sum(dim=[2, 3]) / self.top_k
            u_logits = u_top.topk(self.top_k, dim=3).values.sum(dim=[2, 3]) / self.top_k
        return k_logits, u_logits
    def forward(self, proto: torch.Tensor, kq: torch.Tensor, uq: torch.Tensor, distance: str = "euclidean") -> tuple[torch.Tensor, torch.Tensor]:
        if distance == "euclidean":
            return self._euclidean(proto, kq, uq)
        elif distance == "pixel_sim":
            return self._pixel_similarity(proto, kq, uq)
class EuclideanDistanceCalculator(nn.Module):
    def __init__(self):
        super(EuclideanDistanceCalculator, self).__init__()
    def forward(self, recip_unit, q_feat, o_feat):
        recip = torch.squeeze(recip_unit, dim=0)
        q = torch.squeeze(q_feat, dim=0)
        o = torch.squeeze(o_feat, dim=0)
        dist_recip_q = self._compute_distance(recip, q)
        dist_recip_o = self._compute_distance(recip, o)
        return dist_recip_q.unsqueeze(0), dist_recip_o.unsqueeze(0)
    def _compute_distance(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        feature_dim = a.shape[1]
        a_expanded = a.unsqueeze(1).expand(n, m, feature_dim)  # [n, m, 64]
        b_expanded = b.unsqueeze(0).expand(n, m, feature_dim)  # [n, m, 64]
        distance = torch.sqrt(torch.sum((a_expanded - b_expanded) **2, dim=2))
        return distance.T   
class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))
    def forward(self, supp_center, query_feature):
        supp_center = F.normalize(supp_center, dim=-1)
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1, 2))
        return logits * self.temp