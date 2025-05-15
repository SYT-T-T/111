# IGAN_Scorer_for_MGBOD.py
import torch
import numpy as np
import random
import os
import time

from IGAN_files.network.GANs import Generator, Discriminator
from IGAN_files.optim.trainer import GANsTrainer
from IGAN_files.dataset.tabular import TabularDataset as IGANTabularDataset
from IGAN_files.utils import distribution_sampling
# (可能还需要从 IGAN_files.utils 导入 ROOT_DIR, new_dir, obj_save，如果 TabularDataset 内部严格依赖)

class IGAN_Scorer_for_MGBOD:
    def __init__(self, data_dim: int, latent_dim: int, epochs: int = 50, batch_size: int = 32,
                 lr_d: float = 1e-4, lr_g: float = 1e-4, n_d: int = 1, n_g: int = 1,
                 device: torch.device = torch.device("cpu"), seed: int = 42,
                 view_id: str = "default_view", p_sampling: float = 0.9):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.initial_batch_size = batch_size
        self.lr_d = lr_d; self.lr_g = lr_g
        self.n_d = n_d; self.n_g = n_g
        self.device = device; self.seed = seed
        self.view_id = view_id
        self.p_sampling = p_sampling

        self.igan_trainer = None; self.is_fitted = False
        self.fitted_mean = None # <--- 用于存储 fit 时确定的均值
        self.fitted_std = None  # <--- 用于存储 fit 时确定的标准差
        # self.igan_dataset_instance_for_train = None # 这个可能不需要作为实例变量

    # _prepare_igan_dataset 可以被整合到 fit 中或保持独立
    def _create_igan_tabular_dataset(self, X_np_unscaled: np.ndarray, mode: str = 'train'):
        """辅助函数创建 IGANTabularDataset"""
        dummy_labels = np.zeros(len(X_np_unscaled))
        # 假设 IGANTabularDataset 能够处理 mode='test' 且不强制加载 mean/std 文件
        # 如果 IGANTabularDataset 内部有 results_save_dir, 可能需要传递
        # temp_results_dir = os.path.join(".", "IGAN_files_temp_results", self.view_id, mode) # 示例
        # if not os.path.exists(temp_results_dir): os.makedirs(temp_results_dir)

        dataset_obj = IGANTabularDataset(
            train_data=X_np_unscaled if mode == 'train' else X_np_unscaled, # train_data 是它计算 mean/std 的依据
            train_lab=dummy_labels,
            test_data=X_np_unscaled, # test_data 仅用于结构或在 test mode 下
            test_lab=dummy_labels,
            mode=mode,
            # results_save_dir=temp_results_dir # 可选
        )
        return dataset_obj

    def fit(self, X_train_unscaled_tensor: torch.Tensor):
        if self.seed != -1:
            random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
            if self.device.type == 'cuda': torch.cuda.manual_seed_all(self.seed)

        print(f"  Fitting IGAN_Scorer (view: {self.view_id}) on data of shape {X_train_unscaled_tensor.shape}...")
        X_train_np_unscaled = X_train_unscaled_tensor.cpu().numpy().astype(np.float32)

        if X_train_np_unscaled.shape[0] == 0:
            print(f"Warning: No training samples for IGAN view {self.view_id}, skipping fit.")
            self.is_fitted = False; return

        # 创建 TabularDataset 以进行训练，这将计算并应用标准化
        # TabularDataset 应该将其计算的 mean 和 std 存储到实例属性中
        igan_train_dataset = self._create_igan_tabular_dataset(X_train_np_unscaled, mode='train')

        # 从 TabularDataset 获取计算出的均值和标准差，并保存
        self.fitted_mean = igan_train_dataset.mean
        self.fitted_std = igan_train_dataset.std
        if self.fitted_mean is None or self.fitted_std is None:
            print(f"Critical Warning: Mean/Std not set by IGANTabularDataset for view {self.view_id} during fit. Standardization for predict will fail or be incorrect.")
            # 可以选择在这里强制计算并设置，但这表明 TabularDataset 的修改可能不完整
            if X_train_np_unscaled.shape[0] > 0:
                 self.fitted_mean = np.mean(X_train_np_unscaled, axis=0, dtype=np.float32)
                 self.fitted_std = np.std(X_train_np_unscaled, axis=0, dtype=np.float32)
                 print("   Manually calculated mean/std in IGAN_Scorer fit as fallback.")
            else: # 无法计算，这将导致 predict 失败
                 self.is_fitted = False; return


        current_batch_size = min(self.initial_batch_size, X_train_np_unscaled.shape[0])
        if current_batch_size <= 0: current_batch_size = 1
        print(f"    Using actual batch_size: {current_batch_size} for view {self.view_id}")

        self.igan_trainer = GANsTrainer(
            optimizer_name='adam', lr_d=self.lr_d, lr_g=self.lr_g, epochs=self.epochs,
            batch_size=current_batch_size, device=self.device,
            latent_dim=self.latent_dim, n_d=self.n_d, n_g=self.n_g,
            p_sampling = self.p_sampling
        )
        current_data_dim = X_train_unscaled_tensor.shape[1]
        generator = Generator(input_dim=current_data_dim, output_dim=self.latent_dim)
        discriminator = Discriminator(input_dim=self.latent_dim, output_dim=1)
        self.igan_trainer.build_networks(generator=generator, discriminator=discriminator)

        print(f"  Starting IGAN training for view {self.view_id} ({self.epochs} epochs)...")
        start_time = time.time()
        self.igan_trainer.train(dataset=igan_train_dataset) # TabularDataset 的 train_set 已经是标准化过的
        train_duration = time.time() - start_time
        print(f"  IGAN training finished for view {self.view_id}. Time: {train_duration/60:.2f} Min")
        self.is_fitted = True

    def predict(self, X_test_unscaled_tensor: torch.Tensor) -> torch.tensor:
        if not self.is_fitted or self.igan_trainer is None:
            raise RuntimeError(f"IGAN_Scorer (view: {self.view_id}) must be fitted before predicting.")
        if X_test_unscaled_tensor.shape[0] == 0:
            return torch.tensor([], dtype=torch.float32)

        X_test_np_unscaled = X_test_unscaled_tensor.cpu().numpy().astype(np.float32)

        # 确保使用fit阶段确定的mean和std进行标准化
        std_adjusted = np.where(self.fitted_std < 1e-6, 1e-6, self.fitted_std)
        X_test_np_scaled = (X_test_np_unscaled - self.fitted_mean) / std_adjusted
        # --- 结束标准化 ---

        test_tensor_scaled = torch.from_numpy(X_test_np_scaled).float().to(self.device)

        self.igan_trainer.generator.eval(); self.igan_trainer.discriminator.eval()
        all_scores = []
        predict_batch_size = self.igan_trainer.batch_size if self.igan_trainer and hasattr(self.igan_trainer, 'batch_size') else self.initial_batch_size
        if predict_batch_size <= 0: predict_batch_size = 1
        with torch.no_grad():
            for i in range(0, test_tensor_scaled.size(0), predict_batch_size):
                batch_inputs = test_tensor_scaled[i:i + predict_batch_size]
                transformed_batch = self.igan_trainer.generator(batch_inputs)
                batch_scores = self.igan_trainer.discriminator(transformed_batch)
                all_scores.append(batch_scores.cpu())
        if not all_scores: return torch.tensor([], dtype=torch.float32)
        final_scores_cpu = torch.cat(all_scores).squeeze()
        if final_scores_cpu.ndim == 0: final_scores_cpu = final_scores_cpu.unsqueeze(0)
        return final_scores_cpu