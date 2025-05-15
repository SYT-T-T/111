import numpy as np
import torch
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import traceback
import matplotlib.pyplot as plt  # 加入可视化分数分布

from MGBOD_files.units import load_data as mgbod_load_data
from MGBOD_files.units import get_group_score as mgbod_get_group_score
from MGBOD_files.GB import general_GB as mgbod_general_GB
from IGAN_Scorer_for_MGBOD import IGAN_Scorer_for_MGBOD

try:
    from IGAN_files.utils import split_train_test as igan_original_split_train_test
except ImportError:
    def igan_original_split_train_test(normal_data, abnormal_data, n_train_data=-1, imbalanced=1):
        if n_train_data <= 0 or n_train_data > len(normal_data):
            idx_train_data = list(range(len(normal_data)))
        else:
            idx_train_data = list(range(n_train_data))
        train_data = normal_data[idx_train_data]
        train_lab = np.zeros(len(train_data))
        test_data = np.concatenate([normal_data, abnormal_data])
        test_lab = np.concatenate([np.zeros(len(normal_data)), np.ones(len(abnormal_data))])
        return train_data, train_lab, test_data, test_lab

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 配置参数
igan_l0_params = dict(latent_dim=4, epochs=100, batch_size=32, lr_d=1e-4, lr_g=1e-4, n_d=2, n_g=1, seed=42, p_sampling=0.9)
igan_gb_params = dict(latent_dim=4, epochs=100, batch_size=32, lr_d=1e-4, lr_g=1e-4, n_d=2, n_g=1, seed=42, p_sampling=0.9)
mgbod_gb_params = dict(split1_tolerance=1.0, merge_quantile=0.05, min_gb_size=2, use_kernel_for_merge=True)
k_3wd = 0.7

def main(data_path):
    X_np_raw, y_np_raw = mgbod_load_data(data_path)
    if X_np_raw.ndim != 2 or y_np_raw.ndim != 1:
        raise ValueError(f"Data dimension error: X {X_np_raw.shape}, y {y_np_raw.shape}")
    N = X_np_raw.shape[0]
    y_cpu_raw = torch.from_numpy(y_np_raw.astype(np.float32)).long().cpu()

    normal_idx = np.where(y_np_raw == 0)[0]
    abnormal_idx = np.where(y_np_raw == 1)[0]
    X_normal = X_np_raw[normal_idx]
    X_abnormal = X_np_raw[abnormal_idx]
    l0_train_data, _, _, _ = igan_original_split_train_test(X_normal, X_abnormal, n_train_data=-1)

    igan_l0 = IGAN_Scorer_for_MGBOD(data_dim=X_np_raw.shape[1], device=device, view_id="L0", **igan_l0_params)
    l0_train_tensor = torch.from_numpy(l0_train_data).float().to(device)
    print(f"Training L0 IGAN on {l0_train_tensor.shape[0]} normal samples...")
    igan_l0.fit(X_train_unscaled_tensor=l0_train_tensor)
    X_raw_tensor = torch.from_numpy(X_np_raw).float().to(device)
    print(f"L0 IGAN predicting on all original samples...")
    score_original_cpu = igan_l0.predict(X_test_unscaled_tensor=X_raw_tensor).cpu()

    # ======== IGAN训练/分数分布异常检测代码插入 ========
    print("\n=== IGAN训练集与分数检测 ===")
    print("训练集正常样本数:", l0_train_tensor.shape[0])
    print("全体样本总数:", len(y_np_raw))
    print("异常（1）样本数:", np.sum(y_np_raw == 1))
    print("正常（0）样本数:", np.sum(y_np_raw == 0))
    print("IGAN原始分数统计: max=%.4f min=%.4f mean=%.4f std=%.4f" % (
        np.max(score_original_cpu.numpy()),
        np.min(score_original_cpu.numpy()),
        np.mean(score_original_cpu.numpy()),
        np.std(score_original_cpu.numpy())
    ))
    try:
        plt.figure()
        plt.hist(score_original_cpu.numpy(), bins=50)
        plt.title("IGAN Score Distribution (score_ori)")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.show()
    except Exception as e:
        print("画分数直方图失败：", e)
    # ======== 检测代码结束 ========

    scaler = MinMaxScaler()
    X_np_minmax = scaler.fit_transform(X_np_raw)
    X_tensor_minmax = torch.from_numpy(X_np_minmax).float().to(device)

    score_list_gb_cpu = []
    rep_points_tensor = X_tensor_minmax.clone()
    iteration_count = 0
    while True:
        GB_list, c_cpu, r_cpu = mgbod_general_GB(rep_points_tensor.cpu().float())
        if not c_cpu or len(c_cpu) <= 1:
            print("No more centers, exiting OD_GB_IGAN loop.")
            break
        c_tensor = torch.from_numpy(np.array(c_cpu)).float().to(device)
        igan_gb = IGAN_Scorer_for_MGBOD(data_dim=c_tensor.shape[1], device=device, view_id=f"L_GB_{iteration_count}", **igan_gb_params)
        print(f"Fitting IGAN at GB level {iteration_count} on {c_tensor.shape[0]} centers...")
        igan_gb.fit(X_train_unscaled_tensor=c_tensor)
        center_scores = igan_gb.predict(X_test_unscaled_tensor=c_tensor).cpu()
        mapped_score = mgbod_get_group_score(torch.from_numpy(X_np_minmax).float(), c_cpu, r_cpu, center_scores.numpy())
        score_list_gb_cpu.append(mapped_score.cpu())
        rep_points_tensor = c_tensor
        iteration_count += 1

    def join_scores(score_list_cpu, y_cpu):
        e_list_np = []; weight_list_val = []; processed_scores_cpu = []
        p_outlier = (y_cpu.sum() / len(y_cpu)).item()
        if p_outlier == 0 or p_outlier == 1: p_outlier = 0.1
        for i, score in enumerate(score_list_cpu):
            score = score.clone()
            if score.numel() == 0:
                print(f"Warning: View {i} has empty scores in join_scores. Assigning default entropy/weight.")
                e_list_np.append(np.zeros(len(y_cpu))); weight_list_val.append(0.0); processed_scores_cpu.append(torch.full_like(y_cpu, 0.5)); continue
            if score.min().item() == score.max().item(): score.fill_(0.5)
            else:
                n_s=len(score); n_o=max(1,int(np.ceil(p_outlier*n_s))); n_i=n_s-n_o
                if n_i<=0 or n_o<=0: score.fill_(0.5)
                else:
                    s_idx=torch.argsort(score); t_s_val = score[s_idx[-n_o]].item() if n_o > 0 else score.max().item()
                    m_p=score>=t_s_val; s_p=score[m_p]
                    if s_p.numel()>0 and s_p.max()>s_p.min(): score[m_p]=0.5*(s_p-s_p.min())/(s_p.max()-s_p.min())+0.5
                    elif s_p.numel()>0: score[m_p]=0.75
                    m_n=score<t_s_val; s_n=score[m_n]
                    if s_n.numel()>0 and s_n.max()>s_n.min(): score[m_n]=0.5*(s_n-s_n.min())/(s_n.max()-s_n.min())
                    elif s_n.numel()>0: score[m_n]=0.25
            score=torch.clamp(score,1e-9,1.0-1e-9)
            e=-score*torch.log2(score)-(1-score)*torch.log2(1-score)
            e=e.nan_to_num(0)
            e_list_np.append(e.numpy()); weight_list_val.append((1.0-e.mean().item())); processed_scores_cpu.append(score)
        if not processed_scores_cpu: return torch.tensor([]), [], []
        weights_tensor=torch.tensor(weight_list_val,dtype=torch.float32)
        if weights_tensor.sum().item()>1e-9: norm_weights=weights_tensor/weights_tensor.sum()
        else: norm_weights=torch.ones_like(weights_tensor)/len(weights_tensor) if len(weights_tensor)>0 else torch.tensor([])
        print(f"MS-IGAN Fusion Weights: {norm_weights.numpy()}")
        fused_score=torch.zeros_like(processed_scores_cpu[0])
        for i, w in enumerate(norm_weights): fused_score += w * processed_scores_cpu[i]
        return fused_score, e_list_np, norm_weights

    fused_score_cpu, e_list_np, norm_weights = join_scores([score_original_cpu]+score_list_gb_cpu, y_cpu_raw)

    n_samples = X_np_minmax.shape[0]
    auc_before_svm = roc_auc_score(y_score=fused_score_cpu.numpy(), y_true=y_cpu_raw.numpy()) if n_samples > 1 else np.nan
    sort_indices = torch.argsort(fused_score_cpu)
    p_outlier = (y_cpu_raw.sum() / n_samples).item()
    alpha_rank = min(int(np.ceil(n_samples*(1-p_outlier+k_3wd*p_outlier))), n_samples-1)
    beta_rank = max(int(np.floor(n_samples*(1-p_outlier-k_3wd*(1-p_outlier)))), 0)
    alpha_threshold = fused_score_cpu[sort_indices[alpha_rank]].item()
    beta_threshold = fused_score_cpu[sort_indices[beta_rank]].item()
    if alpha_threshold <= beta_threshold:
        mid = (alpha_threshold + beta_threshold) / 2.0
        beta_threshold = mid - 1e-6; alpha_threshold = mid + 1e-6
        alpha_threshold = min(alpha_threshold, fused_score_cpu.max().item()); beta_threshold = max(beta_threshold, fused_score_cpu.min().item())
    index_ol = torch.where(fused_score_cpu >= alpha_threshold)[0]
    index_il = torch.where(fused_score_cpu <= beta_threshold)[0]
    y_pseudo = torch.zeros_like(y_cpu_raw)
    y_pseudo[index_ol] = 1; y_pseudo[index_il] = 0
    train_idx = torch.cat((index_il, index_ol)).unique()
    X_train_np = X_np_minmax[train_idx.numpy()]
    y_train_np = y_pseudo[train_idx].numpy()
    print(f"Training SVM on {len(train_idx)} reliable samples for fused MS-IGAN...")
    clf = svm.SVC(probability=True, class_weight='balanced', C=1.0)
    clf.fit(X_train_np, y_train_np)
    f_s_p = clf.predict_proba(X_np_minmax)[:,1]
    auc_after_svm = roc_auc_score(y_score=f_s_p, y_true=y_cpu_raw.numpy()) if n_samples > 1 else np.nan
    print(f"Fused MS-IGAN, AUC before SVM: {auc_before_svm:.4f}, AUC after SVM: {auc_after_svm:.4f}")

    auc_ori = roc_auc_score(y_score=score_original_cpu.numpy(), y_true=y_cpu_raw.numpy()) if n_samples > 1 else np.nan
    sort_indices_ori = torch.argsort(score_original_cpu)
    alpha_rank_ori = min(int(np.ceil(n_samples*(1-p_outlier+k_3wd*p_outlier))), n_samples-1)
    beta_rank_ori = max(int(np.floor(n_samples*(1-p_outlier-k_3wd*(1-p_outlier)))), 0)
    alpha_th_ori = score_original_cpu[sort_indices_ori[alpha_rank_ori]].item()
    beta_th_ori = score_original_cpu[sort_indices_ori[beta_rank_ori]].item()
    if alpha_th_ori <= beta_th_ori:
        mid = (alpha_th_ori + beta_th_ori) / 2.0
        beta_th_ori = mid - 1e-6; alpha_th_ori = mid + 1e-6
        alpha_th_ori = min(alpha_th_ori, score_original_cpu.max().item()); beta_th_ori = max(beta_th_ori, score_original_cpu.min().item())
    idx_ol_o = torch.where(score_original_cpu >= alpha_th_ori)[0]
    idx_il_o = torch.where(score_original_cpu <= beta_th_ori)[0]
    y_pseudo_ori = torch.zeros_like(y_cpu_raw)
    y_pseudo_ori[idx_ol_o] = 1; y_pseudo_ori[idx_il_o] = 0
    train_idx_ori = torch.cat((idx_il_o, idx_ol_o)).unique()
    X_train_ori_np = X_np_minmax[train_idx_ori.numpy()]
    y_train_ori_np = y_pseudo_ori[train_idx_ori].numpy()
    print(f"Training SVM for Original IGAN on {len(train_idx_ori)} samples...")
    clf_ori = svm.SVC(probability=True, class_weight='balanced', C=1.0)
    clf_ori.fit(X_train_ori_np, y_train_ori_np)
    f_s_p_o = clf_ori.predict_proba(X_np_minmax)[:,1]
    auc_svm_ori = roc_auc_score(y_score=f_s_p_o, y_true=y_cpu_raw.numpy()) if n_samples > 1 else np.nan
    print(f"AUC on Original IGAN: {auc_ori:.4f}, after SVM: {auc_svm_ori:.4f}")

    return {
        'auc_ori': auc_ori,
        'auc_svm_ori': auc_svm_ori,
        'auc_fused_before_svm': auc_before_svm,
        'auc_fused_after_svm': auc_after_svm,
        'score_ori': score_original_cpu.numpy(),
        'score_fused': fused_score_cpu.numpy(),
        'score_gb_list': [s.numpy() for s in score_list_gb_cpu]
    }

if __name__ == '__main__':
    data_file = "datasets/thyroid.mat"  # 请替换为你实际的数据集路径
    results = main(data_file)
    print(results)