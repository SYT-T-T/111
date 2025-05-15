# MS_IGAN_main.py
import numpy as np
import torch
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn import svm
import traceback
import copy as cp

# --- 从 MGBOD 项目导入 ---
from units import load_data as mgbod_load_data # 重命名以区分
from GB import general_GB # 使用 MGBOD 的粒球生成

# --- 从 IGAN 项目导入 ---
from IGAN_Scorer import IGAN_Scorer # 我们新创建的封装类

# --- 设备定义 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"Using CPU")


# --- 层次化异常分数计算 ---
def calculate_hierarchical_igan_scores(X_original_scaled_device, # 输入已归一化并在 device 上的数据
                                     original_train_data_scaled_np, # 用于训练 L0 IGAN 的归一化正常数据
                                     gb_params, igan_params, max_levels=3):
    """
    构建层次结构并在每层训练/应用 IGAN 计算异常分数。
    Args:
        X_original_scaled_device (torch.Tensor): 原始完整数据集 (已归一化, 在 device 上)。
        original_train_data_scaled_np (np.ndarray): 原始训练数据 (仅正常, 已归一化, CPU Numpy)。
        gb_params (dict): 传递给 general_GB 的参数。
        igan_params (dict): 传递给 IGAN_Scorer 的参数。
        max_levels (int): 最多构建多少层粒球 (L1 到 Lmax)。
    Returns:
        List[torch.Tensor]: 每一层所有原始样本的最终得分列表 (CPU Tensor)。
                           (高层粒球的得分会映射回原始样本)
        List[torch.Tensor]: 每一层中心点的数据 (L0=X, L1=C1, ...)。
    """
    all_level_scores_mapped_to_X = [] # 存储每层映射到原始样本的得分
    all_level_centers = [X_original_scaled_device] # L0 是原始数据

    current_centers_for_gb = X_original_scaled_device # 用于生成下一层粒球
    current_M_for_gb = None

    # --- Level 0: 在原始数据上训练和预测 IGAN ---
    print(f"\n--- Processing Level 0 (Original Data) ---")
    data_dim_l0 = X_original_scaled_device.shape[1]
    igan_scorer_l0 = IGAN_Scorer(data_dim=data_dim_l0, device=device, **igan_params)
    print(f"  Fitting IGAN_L0 on {original_train_data_scaled_np.shape[0]} original training samples...")
    # IGAN_Scorer 的 fit 期望 Tensor, 但其内部会用 TabularDataset，TabularDataset 会标准化
    # 为避免重复标准化，可以修改 IGAN_Scorer 或 TabularDataset
    # 简单起见，这里直接用原始训练数据（已归一化）
    igan_scorer_l0.fit(X_train=torch.from_numpy(original_train_data_scaled_np).float().to(device))
    print(f"  Predicting with IGAN_L0 on {X_original_scaled_device.shape[0]} original samples...")
    scores_l0_cpu = igan_scorer_l0.predict(X_test=X_original_scaled_device) # 返回 CPU Tensor
    all_level_scores_mapped_to_X.append(scores_l0_cpu)
    print(f"Level 0 scores calculated. Shape: {scores_l0_cpu.shape}")

    # --- 更高层级: 基于粒球中心点 ---
    for level in range(1, max_levels + 1):
        print(f"\n--- Processing Level {level} ---")
        print(f"Input centers shape for GB generation: {current_centers_for_gb.shape}")
        if current_centers_for_gb.shape[0] <= igan_params.get('latent_dim', 4) or \
           current_centers_for_gb.shape[0] <= 1: # 至少需要比潜在维度多的点, 且 > 1
            print(f"Stopping hierarchy at level {level}: Not enough centers ({current_centers_for_gb.shape[0]}).")
            break

        # 1. 生成当前层次的粒球中心
        try:
            # general_GB 输入 device tensor, 返回 cpu list
            gb_list, centers_l_cpu, radii_l_cpu = general_GB(current_centers_for_gb, current_M_for_gb, **gb_params)
            print(f"  general_GB finished. Generated {len(centers_l_cpu)} centers for level {level}.")
        except Exception as e: print(f"Error during general_GB at level {level}: {e}"); traceback.print_exc(); break

        if not centers_l_cpu or len(centers_l_cpu) == 0: print(f"Stopping at L{level}: GB returned no centers."); break
        if len(centers_l_cpu) >= current_centers_for_gb.shape[0] and level > 0: print(f"Stopping at L{level}: GB count did not decrease."); break
        if len(centers_l_cpu) <=1: print(f"Stopping at L{level}: Only {len(centers_l_cpu)} center(s)."); break # IGAN至少需要2个点训练

        centers_l_device = torch.tensor(centers_l_cpu, dtype=X_original_scaled_device.dtype).to(device)
        all_level_centers.append(centers_l_device)

        # 2. 在当前层次的中心点上训练和预测 IGAN
        data_dim_level = centers_l_device.shape[1]
        igan_scorer_level = IGAN_Scorer(data_dim=data_dim_level, device=device, **igan_params)
        print(f"  Fitting IGAN_L{level} on {centers_l_device.shape[0]} level {level} centers...")
        igan_scorer_level.fit(X_train=centers_l_device) # 将所有中心点视为正常样本训练

        # 预测所有原始样本在这一层级的得分
        # 需要将原始样本映射到当前层级的中心点，然后用这些中心点的得分作为代理
        # 简化：直接用训练好的 IGAN 对上一层的中心点进行评分，然后映射
        # 或者，更复杂：训练一个映射器将原始点映射到当前层级的中心

        # --- 映射分数回原始样本 ---
        # 方法1：直接使用当前层 IGAN 对所有原始数据评分 (计算量大！)
        # print(f"  Predicting with IGAN_L{level} on {X_original_scaled_device.shape[0]} original samples...")
        # scores_level_mapped_cpu = igan_scorer_level.predict(X_test=X_original_scaled_device)

        # 方法2：获取当前层中心点的 IGAN 得分，然后用 get_group_score 映射
        print(f"  Predicting with IGAN_L{level} on {centers_l_device.shape[0]} level {level} centers...")
        center_scores_level_cpu = igan_scorer_level.predict(X_test=centers_l_device) # CPU tensor

        print(f"  Mapping L{level} scores back to original samples...")
        # get_group_score 输入: data (device), centers (cpu list), radii (cpu list), score (cpu numpy)
        # 返回 numpy array (假设)
        try:
            score_mapped_np = get_group_score(X_original_scaled_device, centers_l_cpu, radii_l_cpu, center_scores_level_cpu.numpy())
            all_level_scores_mapped_to_X.append(torch.from_numpy(score_mapped_np).float()) # CPU Tensor
            print(f"  Level {level} mapped scores calculated. Shape: {score_mapped_np.shape}")
        except Exception as e:
            print(f"Error during get_group_score for level {level}: {e}")
            traceback.print_exc()
            all_level_scores_mapped_to_X.append(torch.zeros(N_original, dtype=torch.float32)) # 占位
            break


        # 3. 准备下一轮输入
        current_centers_for_gb = centers_l_device
        current_M_for_gb = None # 让下一轮的 general_GB 重新计算 M

    if level == max_levels: print(f"Reached maximum levels ({max_levels}).")
    return all_level_scores_mapped_to_X, all_level_centers


# --- 融合与SVM (类似MGBOD，但输入是IGAN分数) ---
def join_and_refine_igan_scores(all_scores_mapped_to_X_cpu, X_original_np_scaled, y_cpu, k_3wd):
    """
    融合多层映射回原始样本的IGAN得分，并进行三支决策+SVM精化。
    Args:
        all_scores_mapped_to_X_cpu (List[torch.Tensor]): 每层映射到原始样本的得分 (CPU Tensor)。
        X_original_np_scaled (np.ndarray): 原始完整数据集 (已归一化, CPU Numpy)。
        y_cpu (torch.Tensor): 原始标签 (CPU Tensor)。
        k_3wd (float): 三支决策参数 Delta。
    Returns:
        float: SVM精化后的AUROC。
        float: 仅融合 (无SVM) 的AUROC。
    """
    if not all_scores_mapped_to_X_cpu: return np.nan, np.nan

    # --- 1. 概率映射、计算熵和视图权重 (类似 MGBOD 的 join) ---
    e_list_np = []; weight_list_val = []; processed_scores_cpu = []
    p_outlier = (y_cpu.sum() / len(y_cpu)).item()
    if p_outlier == 0 or p_outlier == 1: p_outlier = 0.1

    for i in range(len(all_scores_mapped_to_X_cpu)):
        score = all_scores_mapped_to_X_cpu[i].clone() # 已经是 CPU Tensor
        # 概率映射...
        if score.min().item() == score.max().item(): score.fill_(0.5)
        else:
            n_samples = len(score); n_outliers = max(1, int(np.ceil(p_outlier * n_samples)))
            n_inliers = n_samples - n_outliers
            if n_inliers <= 0 or n_outliers <= 0 : score.fill_(0.5)
            else:
                sort_indices = torch.argsort(score); threshold_score = score[sort_indices[-n_outliers]]
                mask_pos = score >= threshold_score; scores_pos = score[mask_pos]
                if scores_pos.numel() > 0 and scores_pos.max() > scores_pos.min(): score[mask_pos] = 0.5*(scores_pos-scores_pos.min())/(scores_pos.max()-scores_pos.min()) + 0.5
                elif scores_pos.numel() > 0: score[mask_pos] = 0.75
                mask_neg = score < threshold_score; scores_neg = score[mask_neg]
                if scores_neg.numel() > 0 and scores_neg.max() > scores_neg.min(): score[mask_neg] = 0.5*(scores_neg-scores_neg.min())/(scores_neg.max()-scores_neg.min())
                elif scores_neg.numel() > 0: score[mask_neg] = 0.25
        score = torch.clamp(score, 1e-9, 1.0 - 1e-9)
        e = - score * torch.log2(score) - (1 - score) * torch.log2(1 - score)
        e = e.nan_to_num(0); e_list_np.append(e.numpy())
        weight_list_val.append((1.0 - e.mean().item())); processed_scores_cpu.append(score)

    # --- 2. 融合 ---
    N_original = X_original_np_scaled.shape[0]
    final_fused_score_cpu = torch.zeros(N_original, dtype=torch.float32)
    if not processed_scores_cpu: return np.nan, np.nan # 如果没有有效分数

    weight_list_tensor = torch.tensor(weight_list_val, dtype=torch.float32)
    if weight_list_tensor.sum().item() > 1e-9: weight_list_normalized = weight_list_tensor / weight_list_tensor.sum()
    else: weight_list_normalized = torch.ones_like(weight_list_tensor) / len(weight_list_tensor) if len(weight_list_tensor)>0 else torch.tensor([])
    print(f"MS-IGAN Fusion Weights (Entropy-based): {weight_list_normalized.numpy()}")
    if processed_scores_cpu and weight_list_normalized.numel() > 0:
        for i in range(len(processed_scores_cpu)):
            final_fused_score_cpu += weight_list_normalized[i] * processed_scores_cpu[i]
    else: return np.nan, np.nan

    auc_before_svm = np.nan
    try: auc_before_svm = roc_auc_score(y_score=final_fused_score_cpu.numpy(), y_true=y_cpu.numpy()); print(f"AUC before SVM (MS-IGAN Fusion): {auc_before_svm:.4f}")
    except ValueError as e: print(f"Could not calculate AUC before SVM: {e}")

    # --- 3. SVM 精化 (类似 MGBOD 的 test) ---
    print("Starting SVM Refinement Phase for MS-IGAN...")
    # ... (三支决策逻辑，使用 final_fused_score_cpu) ...
    # ... (get_uncertainty 计算样本权重 e_svm，使用 e_list_np, weight_list_normalized.numpy().tolist()) ...
    # ... (训练加权 SVM，使用 X_original_np_scaled) ...
    # --- SVM 精化逻辑 (从 MGBOD test 函数中提取并适配) ---
    n_samples = len(final_fused_score_cpu); y_pseudo_cpu = torch.zeros_like(y_cpu)
    auc_after_svm = auc_before_svm # 默认值
    try:
        sort_indices = torch.argsort(final_fused_score_cpu)
        alpha_rank = min(int(np.ceil(n_samples * (1 - p_outlier + k_3wd * p_outlier))), n_samples - 1)
        alpha_threshold = final_fused_score_cpu[sort_indices[alpha_rank]].item()
        beta_rank = max(int(np.floor(n_samples * (1 - p_outlier - k_3wd * (1 - p_outlier)))), 0)
        beta_threshold = final_fused_score_cpu[sort_indices[beta_rank]].item()
        if alpha_threshold <= beta_threshold:
            print(f"Warning: alpha <= beta. Adjusting."); mid = (alpha_threshold + beta_threshold) / 2.0
            beta_threshold = mid - 1e-6; alpha_threshold = mid + 1e-6
            alpha_threshold = min(alpha_threshold, final_fused_score_cpu.max().item()); beta_threshold = max(beta_threshold, final_fused_score_cpu.min().item())
        index_ol_cpu = torch.where(final_fused_score_cpu >= alpha_threshold)[0]; index_il_cpu = torch.where(final_fused_score_cpu <= beta_threshold)[0]

        if len(index_ol_cpu) < 2 or len(index_il_cpu) < 2: print("Warning: Not enough samples for SVM. Skipping."); return auc_before_svm, auc_before_svm

        y_pseudo_cpu[index_ol_cpu] = 1; y_pseudo_cpu[index_il_cpu] = 0
        train_indices_cpu = torch.cat((index_il_cpu, index_ol_cpu)).unique()
        X_train_np = X_original_np_scaled[train_indices_cpu.numpy()]; y_train_np = y_pseudo_cpu[train_indices_cpu].numpy()

        # 计算 SVM 样本权重 (使用 MGBOD 的 get_uncertainty, 需要确保输入兼容)
        # get_uncertainty(index_cpu, e_list_np, weight_list)
        # e_list_np 来自 join_hierarchical，是每层映射到原始样本后的熵
        # weight_list 来自 join_hierarchical，是每层的视图权重
        # 需要一个函数来计算最终融合样本的不确定性，这里简化为均匀权重或使用 MGBOD 的 get_uncertainty
        svm_sample_weights = np.ones(len(train_indices_cpu)) # 简化：均匀权重
        # 如果要用熵权重：
        # svm_sample_weights = get_uncertainty_for_svm(train_indices_cpu, e_list_np, weight_list_normalized.numpy().tolist())

        print(f"Training SVM on {len(train_indices_cpu)} reliable samples...")
        clf = svm.SVC(probability=True, class_weight='balanced', C=1.0);
        clf.fit(X_train_np, y_train_np, sample_weight=svm_sample_weights if 'svm_sample_weights' in locals() and svm_sample_weights is not None else None) # 使用样本权重
        final_scores_proba = clf.predict_proba(X_original_np_scaled)[:, 1]
        auc_after_svm = roc_auc_score(y_score=final_scores_proba, y_true=y_cpu.numpy())
        print(f"AUC after SVM (MS-IGAN): {auc_after_svm:.4f}")
        return auc_after_svm, auc_before_svm
    except (ValueError, IndexError, RuntimeError) as e: print(f"Error during SVM refinement: {e}"); traceback.print_exc(); return auc_before_svm, auc_before_svm


# --- 主执行块 ---
if __name__ == '__main__':
    np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(script_dir, '..', 'datasets') # MGBOD 的数据集路径
    results_dir = os.path.join(script_dir, '..', 'results')
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    files = os.listdir(dir_path)
    results_ms_igan = []; dataset_names = []

    # --- 定义参数 ---
    gb_params = {'split1_tolerance': 1.0, 'merge_quantile': 0.05, 'min_gb_size': 2, 'use_kernel_for_merge': True}
    igan_params = {'latent_dim': 4, 'epochs': 50, 'batch_size': 32, 'lr_d': 1e-4, 'lr_g': 1e-4, 'seed':42} # IGAN 训练参数
    delta_heuristic_params = {'k': 10, 'c': 0.1} # 用于计算 IGAN 的 delta
    k_3wd = 0.7
    max_levels = 3 # 先设小一点，因为训练多个 IGAN 很慢

    print(f"Running Hierarchical Multi-Scale IGAN...")
    print(f"GB Params: {gb_params}"); print(f"IGAN Params: {igan_params}")
    if delta_heuristic_params: print(f"Delta Heuristic: k={delta_heuristic_params['k']}, c={delta_heuristic_params['c']}")
    print(f"Other Params: k_3wd={k_3wd}, max_levels={max_levels}")

    for file in files:
        if not (file.endswith('.npz') or file.endswith('.mat')): continue
        file_path = os.path.join(dir_path, file); print(f"\nProcessing dataset: {file}")
        dataset_names.append(file)
        try:
            X_np_orig, y_np_orig = mgbod_load_data(file_path) # 使用 MGBOD 的加载
            if X_np_orig is None or y_np_orig is None: raise ValueError("Data loading failed.")
            if X_np_orig.ndim != 2 or y_np_orig.ndim != 1: raise ValueError(f"Incorrect dims.")
            if X_np_orig.shape[0] != y_np_orig.shape[0]: raise ValueError("Sample mismatch.")
            if np.isnan(X_np_orig).any() or np.isinf(X_np_orig).any(): X_np_orig = np.nan_to_num(X_np_orig)
            if np.isnan(y_np_orig).any() or np.isinf(y_np_orig).any(): y_np_orig = np.nan_to_num(y_np_orig)

            # --- 数据划分 (类似 IGAN 的 split_train_test) ---
            # 这里简化，假设我们能拿到 MGBOD 原始的 train_data (仅正常)
            # 如果 MGBOD 划分方式不同，需要适配
            # 简单地：从 X_np_orig 中根据 y_np_orig=0 提取正常样本作为训练集
            normal_indices = np.where(y_np_orig == 0)[0]
            abnormal_indices = np.where(y_np_orig == 1)[0]

            if len(normal_indices) == 0: print("Warning: No normal samples found for training IGAN_L0."); continue

            # 为了演示，我们可能需要根据 IGAN 的 n_train_data 来取一部分正常样本训练 L0
            # 假设我们用所有正常样本训练 L0 IGAN
            X_train_normal_np = X_np_orig[normal_indices]
            scaler = MinMaxScaler()
            X_train_normal_np_scaled = scaler.fit_transform(X_train_normal_np) # 用正常数据拟合scaler
            X_all_np_scaled = scaler.transform(X_np_orig) # 用相同的scaler转换所有数据

            X_torch_all_scaled_device = torch.from_numpy(X_all_np_scaled).float().to(device)
            y_torch_cpu = torch.from_numpy(y_np_orig).float() # y 通常在 CPU

            # --- 执行测试 ---
            auc_final, auc_before_svm = join_and_refine_igan_scores(
                # 需要修改 test_MS_KFRAD_Hierarchical 为 test_MS_IGAN_Hierarchical
                # 并调整其内部逻辑以调用 calculate_hierarchical_igan_scores
                # 这里暂时简化，直接调用 calculate_hierarchical_igan_scores 和 join_and_refine_igan_scores
                calculate_hierarchical_igan_scores(
                    X_torch_all_scaled_device, # 传递归一化后的所有数据
                    X_train_normal_np_scaled,  # 传递归一化后的正常训练数据
                    gb_params, igan_params, max_levels
                )[0], # 取 scores list
                X_all_np_scaled, # 用于 SVM
                y_torch_cpu, k_3wd
            )

            results_ms_igan.append([auc_before_svm, auc_final])
            print(f"Results for {file}: AUC_Fusion_Only={auc_before_svm:.4f}, AUC_Fusion+SVM={auc_final:.4f}")

        except Exception as e:
            print(f"!!! Error processing {file}: {e}"); traceback.print_exc()
            results_ms_igan.append([np.nan, np.nan])

    # --- 保存结果 ---
    output_filepath = os.path.join(results_dir, "result_MS_IGAN_Hierarchical.xlsx")
    output_df = pd.DataFrame(data=results_ms_igan, columns=['AUC_Fusion_Only', 'AUC_Fusion+SVM'], index=dataset_names)
    output_df = output_df.sort_index(); output_df.to_excel(output_filepath)
    print(f"\nHierarchical MS-IGAN results saved to {output_filepath}")