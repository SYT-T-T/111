import torch
import numpy as np
import random
import os
import time

from IGAN_files.network.GANs import Generator, Discriminator
from IGAN_files.optim.trainer import GANsTrainer
from IGAN_files.dataset.tabular import TabularDataset as IGANTabularDataset
from IGAN_files.utils import distribution_sampling

class IGAN_Scorer_for_MGBOD:
    def __init__(self, data_dim: int, latent_dim: int, epochs: int = 50, batch_size: int = 32,
                 lr_d: float = 1e-4, lr_g: float = 1e-4, n_d: int = 1, n_g: int = 1,
                 device: torch.device = torch.device("cpu"), seed: int = 42,
                 view_id: str = "default_view", p_sampling: float = 0.9):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.initial_batch_size = batch_size
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.n_d = n_d
        self.n_g = n_g
        self.device = device
        self.seed = seed
        self.view_id = view_id
        self.p_sampling = p_sampling

        self.igan_trainer = None
        self.is_fitted = False
        self.fitted_mean = None
        self.fitted_std = None

    def _create_igan_tabular_dataset(self, X_np_unscaled: np.ndarray, mode: str = 'train'):
        dummy_labels = np.zeros(len(X_np_unscaled))
        dataset_obj = IGANTabularDataset(
            train_data=X_np_unscaled if mode == 'train' else X_np_unscaled,
            train_lab=dummy_labels,
            test_data=X_np_unscaled,
            test_lab=dummy_labels,
            mode=mode,
        )
        return dataset_obj

    def fit(self, X_train_unscaled_tensor: torch.Tensor):
        if self.seed != -1:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        X_np_unscaled = X_train_unscaled_tensor.cpu().numpy()
        self.fitted_mean = X_np_unscaled.mean(axis=0)
        self.fitted_std = X_np_unscaled.std(axis=0) + 1e-8
        X_np_scaled = (X_np_unscaled - self.fitted_mean) / self.fitted_std

        dataset = self._create_igan_tabular_dataset(X_np_scaled, mode='train')
        self.igan_trainer = GANsTrainer(
            lr_d=self.lr_d, lr_g=self.lr_g,
            epochs=self.epochs, batch_size=self.initial_batch_size,
            device=self.device, latent_dim=self.latent_dim,
            n_d=self.n_d, n_g=self.n_g, p_sampling=self.p_sampling
        )
        self.igan_trainer.build_networks(
            generator=Generator(self.data_dim, self.latent_dim),
            discriminator=Discriminator(self.latent_dim, 1)
        )
        self.igan_trainer.train(dataset=dataset)
        # 训练后判别器分布监控
        sample_size = min(64, X_np_scaled.shape[0])
        with torch.no_grad():
            # D_real: 用latent空间的随机向量
            z = torch.randn(sample_size, self.latent_dim).to(self.device)
            D_real = self.igan_trainer.discriminator(z)
            # D_fake: 用原始样本经过G后的latent向量
            sample_idx = np.random.choice(X_np_scaled.shape[0], sample_size, replace=False)
            sample = torch.from_numpy(X_np_scaled[sample_idx]).float().to(self.device)
            fake = self.igan_trainer.generator(sample)
            D_fake = self.igan_trainer.discriminator(fake)
        print(f"[IGAN][{self.view_id}] After train: D_real mean={D_real.mean().item():.4f}, std={D_real.std().item():.4f} | D_fake mean={D_fake.mean().item():.4f}, std={D_fake.std().item():.4f}")
        self.is_fitted = True

    def predict(self, X_test_unscaled_tensor: torch.Tensor):
        assert self.is_fitted, "IGAN_Scorer_for_MGBOD must be fitted before predict!"
        X_np_unscaled = X_test_unscaled_tensor.cpu().numpy()
        # 用fit时的mean/std标准化
        X_np_scaled = (X_np_unscaled - self.fitted_mean) / self.fitted_std
        # 构造dataset（可用mode='test'，实际标签无意义）
        dataset = self._create_igan_tabular_dataset(X_np_scaled, mode='test')
        # 得到判别器分数
        self.igan_trainer.generator.eval()
        self.igan_trainer.discriminator.eval()
        scores = []
        _, test_loader = dataset.loaders(batch_size=self.initial_batch_size, shuffle_test=False, drop_last_test=False)
        with torch.no_grad():
            for batch in test_loader:
                inputs, _, _ = batch
                inputs = inputs.to(self.device)
                transformed_inputs = self.igan_trainer.generator(inputs)
                batch_scores = self.igan_trainer.discriminator(transformed_inputs)
                scores.append(batch_scores.view(-1).cpu())
        # ---- 判别器输出监控（推理期间） ----
        if scores:
            all_scores = torch.cat(scores, dim=0)
            print(f"[IGAN][{self.view_id}] Predict: D(x) mean={all_scores.mean().item():.4f}, std={all_scores.std().item():.4f}")
            return all_scores
        else:
            print(f"[IGAN][{self.view_id}] Predict: empty score, returning 0.5")
            return torch.zeros(X_np_scaled.shape[0]) + 0.5  # fallback