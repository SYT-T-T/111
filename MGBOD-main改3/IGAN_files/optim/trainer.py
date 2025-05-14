__author__ = 'XF'
__date__ = '2023/07/12'


'''
The optmization process of model.
'''
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from ..utils import distribution_sampling
from sklearn.metrics import roc_auc_score,  precision_recall_curve, auc
from torch import optim
import time


class GANsTrainer(object):
    def __init__(self, optimizer_name: str = 'adam', lr_d: float = 1e-4, lr_g: float = 1e-4, epochs: int = 100,
                batch_size: int = 128, latent_dim: int = 4, device: str = 'cuda',
                n_d: int = 1, n_g: int = 1,
                p_sampling: float = 0.9): # <--- 新增采样概率 p

        self.optimizer_name = optimizer_name
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.epochs = epochs
        self.batch_size = batch_size # GANsTrainer 需要保存它用于 DataLoader
        self.latent_dim = latent_dim
        self.device = torch.device(device) # 确保是 torch.device 对象
        self.n_d = n_d         # 判别器每轮训练次数
        self.n_g = n_g         # 生成器每轮训练次数
        self.p_sampling = p_sampling # 保存采样概率

        self.generator = None      # Gθ, 转换器
        self.discriminator = None  # Dφ, 判别器
        self.optimizer_g = None
        self.optimizer_d = None
        self.results = { # 用于存储内部测试结果（如果调用 self.test）
            'AUROC': 0,
            'AUPRC': 0,
        }

    def build_networks(self, generator, discriminator):
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

    def train(self, dataset): # dataset 是 IGANTabularDataset 实例
        if self.generator is None or self.discriminator is None:
            raise ValueError("Generator and Discriminator must be built before training.")

        # Set optimizer
        if self.optimizer_name == 'adam':
            self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr_g)
            self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        elif self.optimizer_name == 'sgd':
            self.optimizer_g = optim.SGD(self.generator.parameters(), lr=self.lr_g)
            self.optimizer_d = optim.SGD(self.discriminator.parameters(), lr=self.lr_d)
        else:
            raise Exception(f'Unknown optimizer name [{self.optimizer_name}].')

        # Get train data loader
        # IGAN 原始代码的 TabularDataset.loaders 使用 drop_last=True for train
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, shuffle_train=True, drop_last_train=True)

        print(f'IGAN Trainer: Start train on {self.device} ===================================')
        self.discriminator.train()
        self.generator.train()
        start_time = time.time()
        flag = True # True: train D, False: train G
        n_d_alternating = 0
        n_g_alternating = 0
        with trange(1, self.epochs + 1, desc=f"IGAN Training Epochs") as pbar:    
            for epoch in trange(1, self.epochs + 1, desc=f"IGAN Training Epochs"):
                g_loss_epoch = 0.0
                d_loss_epoch = 0.0
                n_batches = 0

                if train_loader is None or len(train_loader) == 0:
                    if self.batch_size == 0: print("Error: GANsTrainer effective batch_size is 0."); break
                    print(f"Warning: train_loader empty for epoch {epoch}. Skipping."); continue

                for data_batch in train_loader:
                    # real_samples 是来自 Pdata (真实数据分布) 的样本，已标准化
                    # 我们的目标是将 G(real_samples) 变得像 Pz 的样本
                    real_samples, _, _ = data_batch # 标签和索引在此处不重要
                    real_samples = real_samples.to(self.device)
                    current_batch_actual_size = real_samples.shape[0]
                    if current_batch_actual_size == 0: continue # 跳过空批次

                    # 从目标分布 Pz (截断高斯) 采样 "真实" 的潜在向量
                    target_latent_samples = distribution_sampling(
                        sample_dim=self.latent_dim,
                        size=current_batch_actual_size, # 与当前真实数据批次大小一致
                        p=self.p_sampling
                    ).to(self.device)

                    # 判别器的目标：
                    # D(z_from_Pz) -> 1 (这是来自目标分布的"真实"潜在样本)
                    # D(G(x_from_Pdata)) -> 0 (这是转换器生成的"虚假"潜在样本)
                    labels_for_Pz_samples = torch.ones(current_batch_actual_size, device=self.device)
                    labels_for_Gx_samples = torch.zeros(current_batch_actual_size, device=self.device)

                    # train discriminator
                    if flag:
                        self.optimizer_d.zero_grad()
                        converted_samples_by_G = self.generator(real_samples) # G(x)

                        d_pred_on_target_latent = self.discriminator(target_latent_samples)       # D(z)
                        d_pred_on_converted = self.discriminator(converted_samples_by_G.detach()) # D(G(x)), detach G

                        loss_d_target = F.binary_cross_entropy(d_pred_on_target_latent.squeeze(1), labels_for_Pz_samples)
                        loss_d_converted = F.binary_cross_entropy(d_pred_on_converted.squeeze(1), labels_for_Gx_samples)
                        d_loss = loss_d_target + loss_d_converted

                        d_loss.backward()
                        self.optimizer_d.step()

                        d_loss_epoch += d_loss.item()
                        n_d_alternating += 1
                        if n_d_alternating >= self.n_d:
                            flag = False
                            n_d_alternating = 0
                    # train generator (converter)
                    else: # flag is False
                        self.optimizer_g.zero_grad()
                        converted_samples_by_G = self.generator(real_samples) # G(x)

                        # 生成器的目标: 让 D(G(x)) -> 1 (欺骗判别器，使其认为 G(x) 是来自 Pz 的)
                        g_loss = F.binary_cross_entropy(self.discriminator(converted_samples_by_G).squeeze(1), labels_for_Pz_samples)
                        g_loss.backward()
                        self.optimizer_g.step()

                        g_loss_epoch += g_loss.item()
                        n_g_alternating += 1
                        if n_g_alternating >= self.n_g:
                            flag = True
                            n_g_alternating = 0
                    n_batches += 1

                avg_total_loss = ((g_loss_epoch + d_loss_epoch) / n_batches) if n_batches > 0 else 0.0
                avg_g_loss = (g_loss_epoch / (n_batches / self.n_d if self.n_d > 0 and self.n_g > 0 and n_batches >0 else n_batches if n_batches >0 else 1) ) # 修正可能的除零
                avg_d_loss = (d_loss_epoch / (n_batches / self.n_g if self.n_d > 0 and self.n_g > 0 and n_batches >0 else n_batches if n_batches >0 else 1) ) # 修正可能的除零
                
                pbar.set_description(f'Epoch[{epoch}/{self.epochs}]\t loss: {avg_total_loss:.4f}\t G_loss: {avg_g_loss:.4f}\t D_loss: {avg_d_loss:.4f}')
            # IGAN 原始代码在每轮训练后进行测试，这里我们先注释掉，因为 MS-IGAN 会在所有层级训练完后统一测试
            # if epoch % 1 == 0: # 或者设置一个测试间隔，例如 epoch % 10 == 0
            #     if hasattr(dataset, 'test_set') and dataset.test_set is not None and len(dataset.test_set) > 0 :
            #         print(f'\nEpoch[{epoch}] ################### IGAN Internal testing ######################')
            #         self.test(dataset) # dataset 是 IGANTabularDataset 对象
            #     else:
            #         print(f"\nEpoch[{epoch}] # Skipped IGAN internal test: no test_set in dataset.")


        print(f'IGAN trainer: train time: {(time.time() - start_time) / 60:.2f} Min')
        # IGAN 原始 trainer.train 会返回一些指标，这里可以根据需要添加
        # return {"final_g_loss": avg_g_loss, "final_d_loss": avg_d_loss}


    def test(self, dataset): # dataset 是 IGANTabularDataset 实例, 包含标准化后的测试数据
        if self.generator is None or self.discriminator is None:
            print("Error: Generator or Discriminator not built for IGAN test.")
            return {"AUROC":0.0, "AUPRC":0.0} # 返回默认值

        _, test_loader = dataset.loaders(batch_size=self.batch_size, shuffle_test=False, drop_last_test=False)
        if test_loader is None:
            print("Warning: Test loader is None in IGAN trainer.test().")
            return {"AUROC":0.0, "AUPRC":0.0}


        self.generator.eval(); self.discriminator.eval()
        idx_label_score = []
        with torch.no_grad():
            for data_batch in test_loader:
                inputs, labels, idx = data_batch # inputs 是已标准化的原始特征空间的测试数据
                inputs = inputs.to(self.device)
                # 异常分数是 D(G(inputs))
                transformed_inputs = self.generator(inputs)
                scores = self.discriminator(transformed_inputs)
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        if not idx_label_score:
            print("Warning: No scores generated during IGAN internal test.")
            return {"AUROC":0.0, "AUPRC":0.0}

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels); scores = np.array(scores)

        auroc = 0.0; auprc = 0.0
        if len(np.unique(labels)) > 1: # 确保至少有两个类别才能计算 AUC
            auroc = roc_auc_score(labels, scores)
            print(f'IGAN Internal Test set AUROC: [{auroc * 100:.2f}]%')
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = auc(recall, precision)
            print(f'IGAN Internal Test set AUPRC: [{100. * auprc:.2f}%]')
        else:
            print("IGAN Internal Test set: Only one class present in labels. Cannot compute AUC/AUPRC.")
        self.results['AUROC'] = auroc
        self.results['AUPRC'] = auprc
        return self.results