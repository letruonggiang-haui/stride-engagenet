"""Architectures and augmentation for the landmark-sequence stream, shared by seqmodel.py (inner CV) and
seq_official.py (the gated official opening) so both use identical code."""
import numpy as np, torch, torch.nn as nn
K, D = 4, 223


class AttnPool(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, h, m):
        s = self.w(h).squeeze(-1).masked_fill(~m, -1e4); w = torch.softmax(s, 1); return (h * w.unsqueeze(-1)).sum(1)


class GRUNet(nn.Module):
    def __init__(self, L=300):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3); self.pool = AttnPool(512); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, K))
    def forward(self, x, m):
        h = self.inp(x); lens = m.sum(1).clamp_min(1).cpu()
        p = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False); o, _ = self.gru(p)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1]); return self.head(self.pool(o, m))


class TfmNet(nn.Module):
    def __init__(self, L=300):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(D, 256), nn.LayerNorm(256)); self.pos = nn.Parameter(torch.zeros(1, L, 256)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=0.3, batch_first=True, norm_first=True); self.enc = nn.TransformerEncoder(layer, 4)
        self.norm = nn.LayerNorm(256); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, K))
    def forward(self, x, m):
        h = self.enc(self.inp(x) + self.pos[:, :x.shape[1]], src_key_padding_mask=~m); h = self.norm(h)
        return self.head((h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1))


def augment(x, m):
    """random contiguous temporal crop (80-100 %), frame dropout 10 %, small coordinate noise; all mask-aware."""
    B, L, _ = x.shape; x = x.clone(); m = m.clone()
    for i in range(B):
        n = int(m[i].sum());
        if n < 20: continue
        keep = int(n * np.random.uniform(0.8, 1.0)); s = np.random.randint(0, n - keep + 1)
        seg = x[i, s:s + keep].clone(); x[i] = 0; m[i] = False; x[i, :keep] = seg; m[i, :keep] = True
        drop = torch.rand(keep, device=x.device) < 0.1; m[i, :keep][drop] = False
    return x + 0.02 * torch.randn_like(x) * m.unsqueeze(-1), m


