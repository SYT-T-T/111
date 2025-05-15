# MS_IGAN_MGBOD_main.py
import numpy as np
import torch
import copy as cp
import pandas as pd
import os
import pickle
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import traceback
import time # 添加 time 模块导入，因为 IGAN_Scorer_for_MGBOD 可能会用到

# --- 从 MGBOD 项目导入 ---
from MGBOD_files.units import load_data as mgbod_load_data
from MGBOD_files.units import get_group_score as mgbod_get_group_score
from MGBOD_files.GB import general_GB as mgbod_general_GB
from MGBOD_files.GB import get_newM as mgbod_get_newM

# --- 从 IGAN 项目导入 ---
from IGAN_Scorer_for_MGBOD import IGAN_Scorer_for_MGBOD
try:
    from IGAN_files.utils import split_train_test as igan_original_split_train_test
except ImportError:
    print("CRITICAL ERROR: Could not import 'split_train_test' from IGAN_files.utils.")
    def igan_original_split_train_test(normal_data, abnormal_data, n_train_data=-1, imbalanced=1):
        print("WARNING: Using placeholder for igan_original_split_train_test. Results may be incorrect.")
        if normal_data is None: normal_data = np.array([])
        if abnormal_data is None: abnormal_data = np.array([])
        if len(normal_data) == 0: return np.array([]), np.array([]), abnormal_data, np.ones(len(abnormal_data)) if len(abnormal_data)>0 else np.array([])

        if n_train_data == -1:
            if len(normal_data) > len(abnormal_data): idx_train_start = len(abnormal_data); train_data = normal_data[idx_train_start:]; test_normal_part = normal_data[:idx_train_start]
            else: train_data = normal_data; test_normal_part = np.array([])
        elif n_train_data > 0 and n_train_data <= len(normal_data) : train_data = normal_data[:n_train_data]; test_normal_part = normal_data[n_train_data:]
        else: split_idx = int(len(normal_data) * 0.7); train_data = normal_data[:split_idx]; test_normal_part = normal_data[split_idx:]
        if len(train_data)==0 and len(normal_data)>0: train_data = normal_data[:1] # 保证至少一个训练样本
        train_lab = np.zeros(len(train_data))
        test_data_parts = []
        if len(test_normal_part)>0: test_data_parts.append(test_normal_part)
        if len(abnormal_data)>0: test_data_parts.append(abnormal_data)
        test_data = np.concatenate(test_data_parts) if test_data_parts else np.array([])
        test_lab_parts = []
        if len(test_normal_part)>0: test_lab_parts.append(np.zeros(len(test_normal_part)))
        if len(abnormal_data)>0: test_lab_parts.append(np.ones(len(abnormal_data)))
        test_lab = np.concatenate(test_lab_parts) if test_lab_parts else np.array([])
        return train_data, train_lab, test_data, test_lab

# --- 设备定义 ---
if torch.cuda.is_available():
    device = torch.device("cuda"); print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu"); print(f"Using CPU")

# --- OD_GB_IGAN 函数 ---
def OD_GB_IGAN(X_globally_minmax_scaled_device, y_cpu_for_join,
               igan_params_for_this_gb_view, # 包含 latent_dim, p_sampling 等, 不含 n_train_data
               gb_specific_params):
    if X_globally_minmax_scaled_device.ndim == 1: X_current_centers_device = X_globally_minmax_scaled_device.unsqueeze(0)
    else: X_current_centers_device = cp.deepcopy(X_globally_minmax_scaled_device)
    if X_current_centers_device.shape[0] <= 1: M_current_device = torch.zeros((X_current_centers_device.shape[0], X_current_centers_device.shape[0]), device=device)
    else: M_current_device = torch.cdist(X_current_centers_device, X_current_centers_device, p=2.0)
    score_list_cpu = []; X_true_globally_minmax_scaled_device = cp.deepcopy(X_globally_minmax_scaled_device)
    iteration_count = 0; max_loops_gb = X_globally_minmax_scaled_device.shape[0]
    while iteration_count < max_loops_gb:
        iteration_count += 1; print(f"\n--- OD_GB_IGAN Loop {iteration_count} ---"); print(f"Input centers for GB: {X_current_centers_device.shape}")
        try:
            GB_list, c_cpu, r_cpu = mgbod_general_GB(X_current_centers_device, M_current_device, **gb_specific_params)
            print(f"mgbod_general_GB generated {len(c_cpu)} centers.")
        except Exception as e: print(f"Error in general_GB: {e}"); traceback.print_exc(); break
        if not c_cpu or len(c_cpu)==0: print("Exiting OD_GB_IGAN: No centers."); break
        c_tensor_device = torch.tensor(c_cpu,dtype=X_globally_minmax_scaled_device.dtype).to(device)
        if c_tensor_device.shape[0]<=1: print(f"Skipping IGAN: Only {c_tensor_device.shape[0]} center(s)."); break
        data_dim_level = c_tensor_device.shape[1]
        # current_igan_params 将从 igan_params_for_this_gb_view 直接复制
        igan_scorer_level = IGAN_Scorer_for_MGBOD(data_dim=data_dim_level,device=device,view_id=f"L_GB_{iteration_count}",**igan_params_for_this_gb_view)
        print(f"  Fitting IGAN for L_GB_{iteration_count} on {c_tensor_device.shape[0]} centers...")
        igan_scorer_level.fit(X_train_unscaled_tensor=c_tensor_device)
        print(f"  Predicting with IGAN for L_GB_{iteration_count} on {c_tensor_device.shape[0]} centers...")
        center_scores_level_cpu = igan_scorer_level.predict(X_test_unscaled_tensor=c_tensor_device)
        print("  Mapping scores back to original samples...")
        try:
            score_mapped_device = mgbod_get_group_score(X_true_globally_minmax_scaled_device,c_cpu,r_cpu,center_scores_level_cpu.numpy())
            score_list_cpu.append(score_mapped_device.cpu()); print(f"  Mapped scores added. List length: {len(score_list_cpu)}")
        except Exception as e: print(f"Error during get_group_score: {e}"); traceback.print_exc(); break
        try:
            r_tensor_device = torch.tensor(r_cpu,dtype=X_globally_minmax_scaled_device.dtype).to(device)
            _,M_next_device = mgbod_get_newM(c_tensor_device,r_tensor_device)
        except Exception as e: print(f"Error in get_newM: {e}"); traceback.print_exc(); break
        if M_next_device is None or M_next_device.numel()==0 or M_next_device.max().item()==0: print("Exiting OD_GB_IGAN: M_next condition."); break
        if len(c_cpu)==X_current_centers_device.shape[0] and iteration_count>1: print(f"Warning: GB count stagnant. Forcing exit."); break
        X_current_centers_device = c_tensor_device; M_current_device = M_next_device
    print(f"--- OD_GB_IGAN Finished. Returning {len(score_list_cpu)} score lists. ---")
    return score_list_cpu

# --- run_IGAN_original_view (处理原始尺度数据) ---
def run_IGAN_original_view(X_raw_all_to_predict_device, X_train_normal_raw_for_fit_device,
                           igan_params_for_original_view): # 移除了未使用的 y_device
    print(f"\n--- Processing Original View (L0) with IGAN ---")
    data_dim = X_raw_all_to_predict_device.shape[1]
    igan_scorer_l0 = IGAN_Scorer_for_MGBOD(data_dim=data_dim, device=device, view_id="L0_direct", **igan_params_for_original_view)
    print(f"  Fitting IGAN_L0 on {X_train_normal_raw_for_fit_device.shape[0]} original normal training samples (raw scale)...")
    igan_scorer_l0.fit(X_train_unscaled_tensor=X_train_normal_raw_for_fit_device)
    print(f"  Predicting with IGAN_L0 on {X_raw_all_to_predict_device.shape[0]} original samples (raw scale)...")
    score_cpu = igan_scorer_l0.predict(X_test_unscaled_tensor=X_raw_all_to_predict_device)
    return score_cpu
    

# --- join_scores 函数 (与上一个最终版本相同，处理 CPU Tensors) ---
def join_scores(score_list_cpu, y_cpu):
    # ... (此函数与你提供的 MGBOD `main.py` 中的 `join` 逻辑几乎一致，确保输入输出是 CPU tensor/numpy)
    e_list_np = []; weight_list_val = []; processed_scores_cpu = []
    p_outlier = (y_cpu.sum() / len(y_cpu)).item()
    if p_outlier == 0 or p_outlier == 1: p_outlier = 0.1
    for i in range(len(score_list_cpu)):
        score = score_list_cpu[i].clone()
        if score.numel() == 0 : # 处理空分数列表的情况
             print(f"Warning: View {i} has empty scores in join_scores. Assigning default entropy/weight.")
             e_list_np.append(np.zeros(len(y_cpu))); weight_list_val.append(0.0); processed_scores_cpu.append(torch.full_like(y_cpu, 0.5)); continue
        if score.min().item() == score.max().item(): score.fill_(0.5)
        else:
            n_s=len(score); n_o=max(1,int(np.ceil(p_outlier*n_s))); n_i=n_s-n_o
            if n_i<=0 or n_o<=0: score.fill_(0.5)
            else:
                s_idx=torch.argsort(score); t_s_val = score[s_idx[-n_o]].item() if n_o > 0 else score.max().item() # 处理 n_o 可能为0的情况
                m_p=score>=t_s_val; s_p=score[m_p]
                if s_p.numel()>0 and s_p.max()>s_p.min(): score[m_p]=0.5*(s_p-s_p.min())/(s_p.max()-s_p.min())+0.5
                elif s_p.numel()>0: score[m_p]=0.75
                m_n=score<t_s_val; s_n=score[m_n]
                if s_n.numel()>0 and s_n.max()>s_n.min(): score[m_n]=0.5*(s_n-s_n.min())/(s_n.max()-s_n.min())
                elif s_n.numel()>0: score[m_n]=0.25
        score=torch.clamp(score,1e-9,1.0-1e-9); e=-score*torch.log2(score)-(1-score)*torch.log2(1-score)
        e=e.nan_to_num(0); e_list_np.append(e.numpy()); weight_list_val.append((1.0-e.mean().item())); processed_scores_cpu.append(score)
    
    if not processed_scores_cpu: return torch.tensor([]), [], []

    final_fused_cpu=torch.zeros_like(processed_scores_cpu[0])
    weights_tensor=torch.tensor(weight_list_val,dtype=torch.float32)
    if weights_tensor.sum().item()>1e-9: norm_weights=weights_tensor/weights_tensor.sum()
    else: norm_weights=torch.ones_like(weights_tensor)/len(weights_tensor) if len(weights_tensor)>0 else torch.tensor([])
    print(f"MS-IGAN Fusion Weights: {norm_weights.numpy()}")
    if processed_scores_cpu and norm_weights.numel() > 0 and norm_weights.shape[0] == len(processed_scores_cpu):
        for i in range(len(processed_scores_cpu)): final_fused_cpu += norm_weights[i] * processed_scores_cpu[i]
    return final_fused_cpu, e_list_np, norm_weights.numpy().tolist()

# --- get_uncertainty 函数 (与上一个最终版本相同) ---
def get_uncertainty(index_cpu, e_list_np, weight_list):
    ans_np = np.zeros(len(index_cpu), dtype=np.float32); view_weights_np = np.array(weight_list, dtype=np.float32)
    if isinstance(index_cpu, torch.Tensor): index_np = index_cpu.numpy()
    else: index_np = np.array(index_cpu)
    for i, idx in enumerate(index_np):
        weighted_entropy = 0.0
        for view_idx in range(len(e_list_np)):
            if view_idx < len(e_list_np) and isinstance(e_list_np[view_idx], np.ndarray) and idx < len(e_list_np[view_idx]):
                 entropy_val = e_list_np[view_idx][idx]
                 if view_idx < len(view_weights_np): weighted_entropy += view_weights_np[view_idx] * entropy_val
        ans_np[i] = 1.0 - weighted_entropy
    return ans_np

# --- test_MS_IGAN 函数 (包含完整的 SVM_Ori 计算) ---
def test_MS_IGAN(X_cpu_raw, y_cpu_raw, # 接收原始尺度的 X_cpu_raw, y_cpu_raw
                 igan_params_original_view, # 包含 latent_dim, p_sampling 等 (不应含 n_train_data)
                 igan_params_for_gb_views,  # 包含 latent_dim, p_sampling 等 (不应含 n_train_data)
                 gb_specific_params, k_3wd,
                 n_train_data_for_l0_config): # <--- L0 训练样本数配置
    print("-" * 30); print("Starting MS-IGAN Test"); print("-" * 30)

    # --- 步骤 A: 准备 L0 IGAN 的训练数据 (原始尺度, 使用 IGAN 的 split_train_test) ---
    y_np_raw_labels = y_cpu_raw.numpy()
    X_np_raw_features = X_cpu_raw.numpy() # 这是原始尺度、未经任何变换的数据

    normal_data_np_raw_for_split = X_np_raw_features[y_np_raw_labels == 0]
    abnormal_data_np_raw_for_split = X_np_raw_features[y_np_raw_labels == 1]

    print(f"  Preparing L0 training data using IGAN's split_train_test with n_train_data={n_train_data_for_l0_config}...")
    if len(normal_data_np_raw_for_split) == 0:
        print("Error: No normal samples in input X_cpu_raw for L0 training."); return [np.nan]*4

    temp_abnormal_for_l0_split = abnormal_data_np_raw_for_split
    if n_train_data_for_l0_config == -1 and len(abnormal_data_np_raw_for_split) == 0:
        print("  Warning: n_train_data is -1 for L0 IGAN, but no abnormal samples. Using a dummy abnormal for split_train_test.")
        if len(normal_data_np_raw_for_split) > 0: temp_abnormal_for_l0_split = normal_data_np_raw_for_split[0:1]
        else: print("Fatal: No normal samples at all for L0 split."); return [np.nan]*4
    try:
        l0_train_data_np_raw_for_fit, _, _, _ = igan_original_split_train_test(
            normal_data_np_raw_for_split, temp_abnormal_for_l0_split, n_train_data=n_train_data_for_l0_config
        )
    except Exception as e_split: # 更通用的异常捕获
        print(f"Error/Assertion in igan_original_split_train_test for L0: {e_split}. Defaulting to 70% normal data.")
        split_idx = int(len(normal_data_np_raw_for_split) * 0.7)
        if split_idx == 0 and len(normal_data_np_raw_for_split) > 0: split_idx = 1
        l0_train_data_np_raw_for_fit = normal_data_np_raw_for_split[:split_idx]

    if l0_train_data_np_raw_for_fit.shape[0] == 0: print("Error: No training data for L0 IGAN after split."); return [np.nan]*4
    # X_train_normal_raw_for_l0_device 是原始尺度的、被精确划分出的正常训练数据
    X_train_normal_raw_for_l0_device = torch.from_numpy(l0_train_data_np_raw_for_fit).float().to(device)
    print(f"  L0 IGAN will be trained on {X_train_normal_raw_for_l0_device.shape[0]} raw normal samples.")

    # --- 步骤 B: L0 (原始视图) IGAN 评分 ---
    # run_IGAN_original_view 接收原始尺度的 X (用于 predict) 和原始尺度的正常训练 X (用于 fit)
    score_original_cpu = run_IGAN_original_view(
        X_cpu_raw.to(device), # 预测时用原始尺度的所有数据
        X_train_normal_raw_for_l0_device, # 训练时用精确划分的原始尺度正常数据
        igan_params_original_view # 这个字典不应包含 n_train_data
    )
    if score_original_cpu is None or score_original_cpu.numel() == 0: print("Error: L0 IGAN prediction failed."); return [np.nan]*4

    # --- 步骤 C: 多尺度视图生成和评分 ---
    # 1. 对完整数据 X_cpu_raw 进行一次全局 MinMax 标准化，用于 MGBOD 的粒球生成和后续的 SVM
    scaler_global_for_mgbod_svm = MinMaxScaler()
    X_np_globally_minmax_scaled = scaler_global_for_mgbod_svm.fit_transform(X_cpu_raw.numpy())
    X_globally_minmax_scaled_device = torch.from_numpy(X_np_globally_minmax_scaled).float().to(device)

    # 2. OD_GB_IGAN 在全局标准化数据上操作
    # y_cpu_raw.to(device) 传递给 OD_GB_IGAN，因为 join_scores 在其返回后被调用，需要原始标签的设备版本
    score_list_gb_cpu = OD_GB_IGAN(X_globally_minmax_scaled_device, y_cpu_raw.to(device),
                                   igan_params_for_gb_views, # 这个字典不应包含 n_train_data
                                   gb_specific_params)

    # --- 步骤 D: 融合与精化 ---
    all_scores_cpu = score_list_gb_cpu + [score_original_cpu]
    if not all_scores_cpu or not any(s is not None and s.numel() > 0 for s in all_scores_cpu):print("Error: No valid scores for join."); return [np.nan]*4

    fused_score_cpu, e_list_np, view_weights = join_scores(all_scores_cpu, y_cpu_raw) # join_scores 用原始 y
    if fused_score_cpu.numel() == 0: print("Error: Fusion returned empty scores."); return [np.nan]*4
    auc_before_svm = np.nan
    try: auc_before_svm = roc_auc_score(y_score=fused_score_cpu.numpy(), y_true=y_cpu_raw.numpy()); print(f"AUC before SVM (MS-IGAN Fusion): {auc_before_svm:.4f}")
    except ValueError as e: print(f"Could not calculate AUC before SVM: {e}")

    # SVM 精化 for Fused Scores (使用 X_np_globally_minmax_scaled)
    print("Starting SVM Refinement Phase (for Fused Scores)...")
    p_outlier=(y_cpu_raw.sum()/len(y_cpu_raw)).item();
    if p_outlier==0 or p_outlier==1: p_outlier=0.1
    n_samples = len(fused_score_cpu)
    auc_after_svm_fused = auc_before_svm # Default if SVM fails

    if n_samples > 1 and fused_score_cpu.numel() > 0 :
        y_pseudo_fused_cpu = torch.zeros_like(y_cpu_raw)
        try:
            sort_indices_fused=torch.argsort(fused_score_cpu)
            alpha_rank_fused=min(int(np.ceil(n_samples*(1-p_outlier+k_3wd*p_outlier))),n_samples-1 if n_samples > 0 else 0)
            if n_samples > 0 and sort_indices_fused.numel() > 0 : alpha_th_fused=fused_score_cpu[sort_indices_fused[alpha_rank_fused]].item()
            else: raise ValueError("Empty or invalid fused scores for alpha_f threshold calculation.")
            beta_rank_fused=max(int(np.floor(n_samples*(1-p_outlier-k_3wd*(1-p_outlier)))),0); beta_rank_fused = min(beta_rank_fused, n_samples - 1 if n_samples > 0 else 0)
            if n_samples > 0 and sort_indices_fused.numel() > 0: beta_th_fused=fused_score_cpu[sort_indices_fused[beta_rank_fused]].item()
            else: raise ValueError("Empty or invalid fused scores for beta_f threshold calculation.")

            if alpha_th_fused<=beta_th_fused:
                mid_f=(alpha_th_fused+beta_th_fused)/2.0;beta_th_fused=mid_f-1e-6;alpha_th_fused=mid_f+1e-6
                if fused_score_cpu.numel()>0:alpha_th_fused=min(alpha_th_fused,fused_score_cpu.max().item());beta_th_fused=max(beta_th_fused,fused_score_cpu.min().item())
                else: raise ValueError("Empty fused scores for threshold adjustment")
            idx_ol_f=torch.where(fused_score_cpu>=alpha_th_fused)[0]; idx_il_f=torch.where(fused_score_cpu<=beta_th_fused)[0]

            if len(idx_ol_f)<2 or len(idx_il_f)<2: print("Warning: Not enough samples for Fused SVM. Skipping.")
            else:
                y_pseudo_fused_cpu[idx_ol_f]=1; y_pseudo_fused_cpu[idx_il_f]=0
                train_idx_fused=torch.cat((idx_il_f,idx_ol_f)).unique()
                X_train_fused_np=X_np_globally_minmax_scaled[train_idx_fused.numpy()]
                y_train_fused_np=y_pseudo_fused_cpu[train_idx_fused].numpy()
                svm_weights_fused=get_uncertainty(train_idx_fused, e_list_np, view_weights); svm_weights_fused=np.maximum(svm_weights_fused,1e-6)
                print(f"Training SVM for Fused Scores on {len(train_idx_fused)} samples..."); clf_fused=svm.SVC(probability=True,class_weight='balanced',C=1.0); clf_fused.fit(X_train_fused_np,y_train_fused_np,sample_weight=svm_weights_fused)
                final_scores_proba_fused=clf_fused.predict_proba(X_np_globally_minmax_scaled)[:,1];
                auc_after_svm_fused = roc_auc_score(y_score=final_scores_proba_fused,y_true=y_cpu_raw.numpy()); print(f"AUC after SVM (Fused Scores): {auc_after_svm_fused:.4f}")
        except Exception as e_svm_fused: print(f"Error in Fused SVM refinement: {e_svm_fused}"); traceback.print_exc()
    else: print("Warning: Not enough samples in fused_score_cpu for SVM refinement. Skipping.")

    # --- AUC Original View Only (IGAN score) ---
    auc_ori_only_igan = np.nan
    if score_original_cpu is not None and score_original_cpu.numel() > 0:
        try: auc_ori_only_igan = roc_auc_score(y_score=score_original_cpu.numpy(), y_true=y_cpu_raw.numpy()); print(f"AUC Original View Only (IGAN score): {auc_ori_only_igan:.4f}")
        except ValueError as e: print(f"Could not calculate AUC for Original IGAN score: {e}")

    # --- SVM 精化 for Original View Score ---
    auc_svm_ori = auc_ori_only_igan # Default if SVM fails
    if score_original_cpu is not None and score_original_cpu.numel() > 0 and n_samples > 1 :
        print("\nCalculating SVM on Original View IGAN scores...")
        y_pseudo_ori_cpu = torch.zeros_like(y_cpu_raw)
        try:
            sort_indices_ori=torch.argsort(score_original_cpu)
            alpha_rank_ori=min(int(np.ceil(n_samples*(1-p_outlier+k_3wd*p_outlier))),n_samples-1 if n_samples>0 else 0)
            if n_samples > 0 and sort_indices_ori.numel() > 0: alpha_th_ori=score_original_cpu[sort_indices_ori[alpha_rank_ori]].item()
            else: raise ValueError("Empty original scores for alpha_o sort")
            beta_rank_ori=max(int(np.floor(n_samples*(1-p_outlier-k_3wd*(1-p_outlier)))),0); beta_rank_ori=min(beta_rank_ori, n_samples-1 if n_samples>0 else 0)
            if n_samples > 0 and sort_indices_ori.numel() > 0: beta_th_ori=score_original_cpu[sort_indices_ori[beta_rank_ori]].item()
            else: raise ValueError("Empty original scores for beta_o sort")

            if alpha_th_ori<=beta_th_ori:mid_o=(alpha_th_ori+beta_th_ori)/2.0;beta_th_ori=mid_o-1e-6;alpha_th_ori=mid_o+1e-6;
            if score_original_cpu.numel()>0 : alpha_th_ori=min(alpha_th_ori,score_original_cpu.max().item());beta_th_ori=max(beta_th_ori,score_original_cpu.min().item())
            else: raise ValueError("Empty original scores for threshold adj")
            idx_ol_o=torch.where(score_original_cpu>=alpha_th_ori)[0];idx_il_o=torch.where(score_original_cpu<=beta_th_ori)[0]
            if len(idx_ol_o)<2 or len(idx_il_o)<2:print("Warning: Not enough samples for Original View SVM.")
            else:
                y_pseudo_ori_cpu[idx_ol_o]=1;y_pseudo_ori_cpu[idx_il_o]=0
                train_idx_ori_cpu=torch.cat((idx_il_o,idx_ol_o)).unique()
                X_train_ori_np=X_np_globally_minmax_scaled[train_idx_ori_cpu.numpy()] # 使用全局标准化数据
                y_train_ori_np=y_pseudo_ori_cpu[train_idx_ori_cpu].numpy()
                print(f"Training SVM for Original View on {len(train_idx_ori_cpu)} samples (Uniform Weights)...");clf_o=svm.SVC(probability=True,class_weight='balanced',C=1.0);clf_o.fit(X_train_ori_np,y_train_ori_np)
                f_s_p_o=clf_o.predict_proba(X_np_globally_minmax_scaled)[:,1] # 使用全局标准化数据
                auc_svm_ori = roc_auc_score(y_score=f_s_p_o,y_true=y_cpu_raw.numpy());print(f"AUC after SVM (Original View Only): {auc_svm_ori:.4f}")
        except Exception as e_svm_o:print(f"Error during SVM for Original View: {e_svm_o}");traceback.print_exc()
    else: print("Warning: Not enough samples in score_original_cpu for Original View SVM. Skipping.")

    return auc_ori_only_igan, auc_before_svm, auc_svm_ori, auc_after_svm_fused

# --- 主执行块 (`if __name__ == '__main__':`) ---
if __name__ == '__main__':
    np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(script_dir, '.', 'datasets')
    results_dir = os.path.join(script_dir, '.', 'results')
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    files = os.listdir(dir_path)
    results_ms_igan = []; dataset_names = []

    gb_specific_params = {}
    igan_base_params = {'epochs': 100, 'batch_size': 32, 'lr_d': 1e-4, 'lr_g': 1e-4, 'seed':42, 'n_d':1, 'n_g':1, 'p_sampling': 0.9}
    # n_train_data 不再是 igan_base_params 的一部分，将单独处理
    k_3wd = 0.7

    latent_dim_config_map = {
        (0, 20):    {'original_view': 4, 'gb_view': 4}, (20, 50):   {'original_view': 8, 'gb_view': 8},
        (50, 100):  {'original_view': 16, 'gb_view': 16},(100, 300): {'original_view': 32, 'gb_view': 32},
        (300, float('inf')): {'original_view': 64, 'gb_view': 64},
        'default_original': 8, 'default_gb': 8
    }
    # --- 新增：为 L0 的 split_train_test 设置 n_train_data ---
    # 这个值应该与原始 IGAN 项目运行时使用的值一致 (通常是默认 -1)
    n_train_data_for_l0_config = -1
    # ---

    print(f"Running Multi-Scale IGAN (MGBOD framework with Range-based Latent Dim)...")
    print(f"GB Params: (Using MGBOD original GB.py)"); print(f"IGAN Base Params: {igan_base_params}")
    print(f"Latent Dim Config Map: {latent_dim_config_map}"); print(f"Other Params: k_3wd={k_3wd}, L0_n_train_data={n_train_data_for_l0_config}")

    for file in files:
        if not (file.endswith('.npz') or file.endswith('.mat')): continue
        file_path = os.path.join(dir_path, file); print(f"\nProcessing dataset: {file}"); dataset_names.append(file)
        try:
            X_np, y_np = mgbod_load_data(file_path)
            if X_np is None or y_np is None: raise ValueError("Data loading failed.")
            if X_np.ndim!=2 or y_np.ndim!=1: raise ValueError(f"Incorrect dims: X={X_np.shape}, y={y_np.shape}")
            if X_np.shape[0]!=y_np.shape[0]: raise ValueError("Sample mismatch.")
            if np.isnan(X_np).any() or np.isinf(X_np).any():X_np=np.nan_to_num(X_np)
            if np.isnan(y_np).any() or np.isinf(y_np).any():y_np=np.nan_to_num(y_np)
            X_torch_cpu=torch.from_numpy(X_np).float();y_torch_cpu=torch.from_numpy(y_np).float();original_data_dim=X_torch_cpu.shape[1]

            selected_latent_dim_original=latent_dim_config_map.get('default_original',4)
            selected_latent_dim_gb=latent_dim_config_map.get('default_gb',4)
            sorted_ranges=sorted([k for k in latent_dim_config_map if isinstance(k,tuple)],key=lambda x:x[0])
            for(min_dim,max_dim)in sorted_ranges:
                if min_dim<=original_data_dim<max_dim:
                    selected_latent_dim_original=latent_dim_config_map[(min_dim,max_dim)].get('original_view',selected_latent_dim_original)
                    selected_latent_dim_gb=latent_dim_config_map[(min_dim,max_dim)].get('gb_view',selected_latent_dim_gb)
                    break
            print(f"  Selected L0 latent_dim: {selected_latent_dim_original}, L1+ latent_dim: {selected_latent_dim_gb} (Orig Dim: {original_data_dim})")

            current_igan_params_original=igan_base_params.copy()
            current_igan_params_original['latent_dim']=selected_latent_dim_original
            # 如果需要，在这里将 n_train_data_for_l0_config 添加到 current_igan_params_original
            # 但更推荐直接传递给 test_MS_IGAN
            current_igan_params_gb=igan_base_params.copy()
            current_igan_params_gb['latent_dim']=selected_latent_dim_gb

            auc_igan_ori,auc_fused,auc_svm_ori,auc_svm_fused=test_MS_IGAN(
                X_torch_cpu, y_torch_cpu,
                current_igan_params_original, # IGAN 参数 for L0
                current_igan_params_gb,       # IGAN 参数 for GB views
                gb_specific_params, k_3wd,
                n_train_data_for_l0_config # <--- 传递 n_train_data for L0
            )
            results_ms_igan.append([auc_igan_ori,auc_fused,auc_svm_ori,auc_svm_fused])
            print(f"Results for {file}: AUC_IGAN_Ori={auc_igan_ori:.4f}, AUC_Fused={auc_fused:.4f}, AUC_SVM_Ori={auc_svm_ori:.4f}, AUC_SVM_Fused={auc_svm_fused:.4f}")
        except Exception as e:print(f"!!! Error processing {file}: {e}");traceback.print_exc();results_ms_igan.append([np.nan]*4)

    output_filepath=os.path.join(results_dir,"result_MS_IGAN_MGBOD_final_v5.xlsx") # 更新文件名
    output_df=pd.DataFrame(data=results_ms_igan,columns=['AUC_IGAN_Ori','AUC_Fused_Only','AUC_SVM_Ori','AUC_SVM_Fused'],index=dataset_names)
    output_df=output_df.sort_index();output_df.to_excel(output_filepath)
    print(f"\nMS-IGAN results saved to {output_filepath}")