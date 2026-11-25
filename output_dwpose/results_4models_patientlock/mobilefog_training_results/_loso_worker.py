import json
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lb6_graph_A() -> np.ndarray:
    V = 6
    self_link = np.eye(V, dtype=np.float32)
    inward_edges = [(2, 0), (3, 1), (4, 2), (5, 3), (0, 1)]
    inward = np.zeros((V, V), dtype=np.float32)
    outward = np.zeros((V, V), dtype=np.float32)
    for i, j in inward_edges:
        inward[j, i] = 1.0
        outward[i, j] = 1.0

    def norm_adj(M):
        deg = M.sum(axis=0, keepdims=True)
        deg[deg == 0] = 1.0
        return M / deg

    return np.stack([norm_adj(self_link), norm_adj(inward), norm_adj(outward)], axis=0)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, A: np.ndarray):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        K, V, _ = A.shape
        self.K = K
        self.edge_mask = nn.Parameter(torch.ones(K, V, V))
        self.conv = nn.Conv2d(in_c, out_c * K, kernel_size=1)

    def forward(self, x):
        B, _, T, V = x.shape
        x = self.conv(x).view(B, self.K, -1, T, V)
        A_eff = self.A * self.edge_mask
        return torch.einsum("bkctv,kvw->bctw", x, A_eff).contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, A: np.ndarray, stride: int = 1, dropout: float = 0.25):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_c, out_c, A)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.tconv_dw = nn.Conv2d(out_c, out_c, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0), groups=out_c)
        self.tconv_pw = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(dropout)
        if in_c != out_c or stride != 1:
            self.residual = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_c))
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        h = F.relu(self.bn1(self.sgcn(x)))
        h = self.bn2(self.tconv_pw(self.tconv_dw(h)))
        return F.relu(self.drop(h) + res)


class LightweightSTGCN(nn.Module):
    def __init__(self, in_c: int, A: np.ndarray, dropout: float = 0.25):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_c * A.shape[1])
        self.blocks = nn.Sequential(
            STGCNBlock(in_c, 32, A, stride=1, dropout=dropout),
            STGCNBlock(32, 64, A, stride=2, dropout=dropout),
            STGCNBlock(64, 64, A, stride=2, dropout=dropout),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x).view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        return self.head(self.blocks(x).mean(dim=(2, 3))).squeeze(-1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SelectiveSSM1D(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_inner + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u):
        B, L, D = u.shape
        x = F.silu(self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
        dt_in, B_t, C_t = torch.split(self.x_proj(x), [D, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_in))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            h = torch.exp(dt_t * A.unsqueeze(0)) * h + dt_t * B_t[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1)
            ys.append((h * C_t[:, t, :].unsqueeze(1)).sum(dim=-1))
        return torch.stack(ys, dim=1) + x * self.D.unsqueeze(0).unsqueeze(0)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int = 64, expand: int = 2, d_state: int = 16, dropout: float = 0.2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.ssm_fwd = SelectiveSSM1D(d_inner, d_state=d_state)
        self.ssm_bwd = SelectiveSSM1D(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        gate = F.silu(z)
        y_fwd = self.ssm_fwd(u) * gate
        y_bwd = torch.flip(self.ssm_bwd(torch.flip(u, dims=[1])), dims=[1]) * gate
        return x + self.drop(self.out_proj(y_fwd + y_bwd))


class PureBiMamba(nn.Module):
    def __init__(self, in_c: int, num_joints: int = 6, d_model: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_c * num_joints, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.layers = nn.ModuleList([BiMambaBlock(d_model=d_model, expand=2, d_state=16, dropout=dropout) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, C, T, V = x.shape
        h = self.stem(x.permute(0, 2, 1, 3).contiguous().view(B, T, C * V))
        for blk in self.layers:
            h = blk(h)
        return self.head(self.norm_f(h).mean(dim=1)).squeeze(-1)


class HybridGCNBiMamba(nn.Module):
    def __init__(self, in_c: int, A: np.ndarray, d_model: int = 64, n_mamba: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = SpatialGraphConv(in_c, 32, A)
        self.bn1 = nn.BatchNorm2d(32)
        self.gcn2 = SpatialGraphConv(32, d_model, A)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.drop_g = nn.Dropout(dropout)
        self.mamba_layers = nn.ModuleList([BiMambaBlock(d_model=d_model, expand=2, d_state=16, dropout=dropout) for _ in range(n_mamba)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = F.silu(self.bn1(self.gcn1(x)))
        h = self.drop_g(F.silu(self.bn2(self.gcn2(h))))
        h = h.mean(dim=-1).transpose(1, 2).contiguous()
        for blk in self.mamba_layers:
            h = blk(h)
        return self.head(self.norm_f(h).mean(dim=1)).squeeze(-1)


def extract_fft_features(X_raw: np.ndarray, fs: float) -> np.ndarray:
    N, C, T, V = X_raw.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_mask = (freqs > 3.0) & (freqs <= min(8.0, 0.49 * fs))
    feats = []
    for c_idx in [0, 1, 3, 4]:
        arr = X_raw[:, c_idx, :, :]
        feats.extend([arr.mean(axis=1), arr.std(axis=1), np.max(np.abs(arr), axis=1)])
    for c_idx in [3, 4]:
        sig = X_raw[:, c_idx, :, :] - X_raw[:, c_idx, :, :].mean(axis=1, keepdims=True)
        psd = np.abs(np.fft.rfft(sig, axis=1)) ** 2
        e_loco = psd[:, loco_mask, :].sum(axis=1) + 1e-8
        e_freeze = psd[:, freeze_mask, :].sum(axis=1) + 1e-8
        feats.extend([np.log1p(e_loco), np.log1p(e_freeze), np.log1p(e_freeze / e_loco)])
    for l_j, r_j in [(0, 1), (2, 3), (4, 5)]:
        dist_xy = np.sqrt((X_raw[:, 0, :, l_j] - X_raw[:, 0, :, r_j]) ** 2 + (X_raw[:, 1, :, l_j] - X_raw[:, 1, :, r_j]) ** 2)
        asym_v = np.abs(X_raw[:, 3, :, l_j] - X_raw[:, 3, :, r_j]) + np.abs(X_raw[:, 4, :, l_j] - X_raw[:, 4, :, r_j])
        feats.extend([dist_xy.mean(axis=1, keepdims=True), dist_xy.std(axis=1, keepdims=True), asym_v.mean(axis=1, keepdims=True), asym_v.std(axis=1, keepdims=True)])
    feats.append(X_raw[:, 2, :, :].mean(axis=1))
    return np.nan_to_num(np.concatenate(feats, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def find_best_thr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.15, 0.85, 71):
        sc = f1_score(y_true, (y_prob >= thr).astype(np.int64), zero_division=0)
        if sc > best_f1:
            best_f1, best_thr = sc, float(thr)
    return best_thr


def fit_torch_fold(model, X_tr, y_tr, X_va, y_va, X_te, cfg, device):
    model = model.to(device)
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))), batch_size=cfg["batch_size"], shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va)), batch_size=128, shuffle=False)
    dl_te = DataLoader(TensorDataset(torch.from_numpy(X_te)), batch_size=128, shuffle=False)

    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    @torch.no_grad()
    def pred(dl):
        model.eval()
        out = []
        for (xb,) in dl:
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        return np.concatenate(out)

    best_pr, best_state, bad = -1.0, None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()
        p_va = pred(dl_va)
        pr = average_precision_score(y_va, p_va) if len(np.unique(y_va)) > 1 else 0.0
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    p_va = pred(dl_va)
    p_te = pred(dl_te)
    thr = find_best_thr(y_va, p_va)
    return p_te.astype(np.float32), float(thr)


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    gpu_label = job["gpu_label"]
    cfg = job["cfg"]
    fold_sids = job["fold_sids"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A_LB6 = build_lb6_graph_A()

    # Gabungkan X MENTAH (bukan X_scaled!) dari train + val + test
    sdir = Path(cfg["scen_dir"])
    Xs, ys, sids, tids = [], [], [], []
    for sp in ["train", "val", "test"]:
        d = np.load(sdir / f"{sp}.npz", allow_pickle=True)
        Xs.append(d["X"].astype(np.float32))
        ys.append(d[cfg["label_key"]].astype(np.int64))
        sids.append(d["subject_id"].astype(str))
        tids.append(d["trial_id"].astype(str))
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    sid_all = np.concatenate(sids, axis=0)
    tid_all = np.concatenate(tids, axis=0)
    F_all = extract_fft_features(X_all, fs=float(cfg["hz"]))

    unique_sids = sorted(np.unique(sid_all).tolist())
    # Kelompokkan subjek FoG aktif vs kontrol 0% FoG untuk stratifikasi inner-val
    active_sids = [s for s in unique_sids if y_all[sid_all == s].sum() > 0]
    ctrl_sids = [s for s in unique_sids if y_all[sid_all == s].sum() == 0]
    in_c = int(X_all.shape[1])

    for test_sid in fold_sids:
        fold_file = out_dir / f"fold_{test_sid}.npz"
        if fold_file.exists():
            continue
        t0 = time.time()
        fold_idx = unique_sids.index(test_sid)
        rng = np.random.default_rng(cfg["seed"] + fold_idx)

        rem_active = [s for s in active_sids if s != test_sid]
        rem_ctrl = [s for s in ctrl_sids if s != test_sid]
        n_val = cfg["inner_val_subjects"]
        n_val_ctrl = min(1, len(rem_ctrl))
        n_val_act = min(n_val - n_val_ctrl, len(rem_active))
        val_sids = set(rng.choice(rem_active, size=n_val_act, replace=False).tolist()) | set(rng.choice(rem_ctrl, size=n_val_ctrl, replace=False).tolist())

        te_mask = sid_all == test_sid
        va_mask = np.isin(sid_all, list(val_sids))
        tr_mask = ~(te_mask | va_mask)

        # ZERO DATA LEAKAGE: Fit StandardScaler HANYA di X_tr fold ini!
        X_tr_raw, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_va_raw, y_va = X_all[va_mask], y_all[va_mask]
        X_te_raw, y_te = X_all[te_mask], y_all[te_mask]

        mean = X_tr_raw.mean(axis=(0, 2, 3), keepdims=True)
        std = X_tr_raw.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        X_tr = ((X_tr_raw - mean) / std).astype(np.float32)
        X_va = ((X_va_raw - mean) / std).astype(np.float32)
        X_te = ((X_te_raw - mean) / std).astype(np.float32)

        # 1. LightGBM + FFT
        seed_everything(cfg["seed"] + fold_idx)
        pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.8, subsample=0.8, scale_pos_weight=pos_w,
            random_state=cfg["seed"] + fold_idx, n_jobs=-1, verbose=-1
        )
        clf.fit(F_all[tr_mask], y_tr, eval_set=[(F_all[va_mask], y_va)], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(40, verbose=False)])
        p_va_lgb = clf.predict_proba(F_all[va_mask])[:, 1]
        p_te_lgb = clf.predict_proba(F_all[te_mask])[:, 1].astype(np.float32)
        thr_lgb = find_best_thr(y_va, p_va_lgb)

        # 2. Light ST-GCN-cls
        seed_everything(cfg["seed"] + fold_idx)
        p_te_stgcn, thr_stgcn = fit_torch_fold(LightweightSTGCN(in_c, A_LB6), X_tr, y_tr, X_va, y_va, X_te, cfg, device)

        # 3. Pure Bi-Mamba
        seed_everything(cfg["seed"] + fold_idx)
        p_te_mamba, thr_mamba = fit_torch_fold(PureBiMamba(in_c, 6, 64, 2), X_tr, y_tr, X_va, y_va, X_te, cfg, device)

        # 4. Hybrid GCN + Bi-Mamba
        seed_everything(cfg["seed"] + fold_idx)
        p_te_hyb, thr_hyb = fit_torch_fold(HybridGCNBiMamba(in_c, A_LB6, 64, 2), X_tr, y_tr, X_va, y_va, X_te, cfg, device)

        np.savez_compressed(
            fold_file,
            test_sid=np.array(test_sid),
            indices=np.where(te_mask)[0].astype(np.int64),
            y_te=y_te,
            trial_id=tid_all[te_mask],
            p_lgbm=p_te_lgb, thr_lgbm=np.float32(thr_lgb),
            p_stgcn=p_te_stgcn, thr_stgcn=np.float32(thr_stgcn),
            p_mamba=p_te_mamba, thr_mamba=np.float32(thr_mamba),
            p_hybrid=p_te_hyb, thr_hybrid=np.float32(thr_hyb),
        )
        print(f"[GPU {gpu_label}] LOSO Fold Subjek {test_sid} selesai ({len(y_te)} win, FoG={y_te.mean()*100:.1f}%) dalam {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
