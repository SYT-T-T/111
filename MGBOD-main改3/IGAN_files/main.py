__author__ = 'XF'
__date__ = '2023/07/12'

'''
the entrance of the project.
'''

from os import path as osp
import os
import click
import torch
import numpy as np
import random
import glob
import pandas as pd
import traceback

from network.GANs import Discriminator, Generator
from optim.trainer import GANsTrainer
from dataset.tabular import TabularDataset
from dataset.load_data import load_data
from utils import split_train_test


ROOT_DIR = osp.dirname(osp.abspath(__file__))

@click.command()
@click.option('--data_dir', type=str, default='dataset', help='数据集目录路径')
@click.option('--optimizer_name', type=str, default='adam', help='')
@click.option('--epochs', type=int, default=100, help='The total iteration number.')
@click.option('--batch_size', type=int, default=32, help='')
@click.option('--lr_d', type=float, default=1e-4, help='the learning rate of discriminator.')
@click.option('--lr_g', type=float, default=1e-4, help='the learning rate of generator.')
@click.option('--latent_dim', type=int, default=4, help='the data dimension in latent space.')
@click.option('--seed', type=int, default=-1, help='the random seed.')
@click.option('--n_train_data', type=int, default=-1, help='the sample size of training data.')
@click.option('--train', type=bool, default=True, help='when it is True, training mode, otherwise testing mode.')
@click.option('--n_d', type=int, default=1, help='The updating time of discriminator in a circle.')
@click.option('--n_g', type=int, default=1, help='The updating time of generator in a circle.')
@click.option('--repeat', type=int, default=1, help='The repeat time of the training process.')
@click.option('--process_all', type=bool, default=False, help='是否处理目录下所有数据集文件')

def main(data_dir, optimizer_name, epochs, batch_size, lr_d, lr_g, latent_dim, seed, n_train_data, train, n_d, n_g, repeat, process_all):
    
    print(f'=============================== IGAN ====================================== ')
    
    # 设置设备
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Computation device: {device}')
    
    # 实验设置
    print('experiment settings')
    print(f'Optimizer: {optimizer_name}')
    print(f'Epochs: [{epochs}]')
    print(f'Batch size: [{batch_size}]')
    print(f'Latent dimension: [{latent_dim}]')
    print(f'Learning rate of discriminator: [{lr_d}]')
    print(f'Learning rate of genertor: [{lr_g}]')
    
    # 为不同潜在维度配置映射
    latent_dim_config_map = {
        'thyroid': 4,
        'arrhythmia': 6,
        'Cardio': 8,
        'adult': 4,
        'abalone': 4,
        'Speech': 6,
        'Vowels': 6,
        'Vertebral': 4,
        'Satellite': 8
    }
    
    if process_all:
        # 处理目录下所有数据集文件
        dir_path = os.path.join(ROOT_DIR, data_dir)
        files = os.listdir(dir_path)
        dataset_names = []
        results_ms_igan = []
        
        # 创建结果目录
        results_dir = os.path.join(ROOT_DIR, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for file in files:
            # 只处理.mat和.npz文件
            if not (file.endswith('.mat') or file.endswith('.npz')):
                continue
                
            file_path = os.path.join(dir_path, file)
            print(f"\nProcessing dataset: {file}")
            dataset_name = os.path.splitext(file)[0]
            dataset_names.append(dataset_name)
            
            try:
                # 加载数据
                normal_data, abnormal_data = None, None
                try:
                    # 尝试使用dataset名称加载
                    normal_data, abnormal_data = load_data(dataset=dataset_name)
                except Exception:
                    # 如果失败，直接使用文件路径加载
                    from scipy.io import loadmat
                    data = loadmat(file_path)
                    if 'X' in data and 'y' in data:
                        normal_data = data['X'][data['y'][:, 0] == 0]
                        abnormal_data = data['X'][data['y'][:, 0] == 1]
                    else:
                        raise ValueError(f"数据文件 {file} 格式不正确，需要包含'X'和'y'键")
                
                if normal_data is None or abnormal_data is None:
                    raise ValueError("数据加载失败")
                    
                if normal_data.ndim != 2 or abnormal_data.ndim != 2:
                    raise ValueError(f"数据维度不正确")
                    
                # 数据预处理
                if np.isnan(normal_data).any() or np.isinf(normal_data).any():
                    normal_data = np.nan_to_num(normal_data)
                if np.isnan(abnormal_data).any() or np.isinf(abnormal_data).any():
                    abnormal_data = np.nan_to_num(abnormal_data)
                
                # 获取当前数据集的最佳潜在维度
                current_latent_dim = latent_dim_config_map.get(dataset_name, latent_dim)
                
                # 训练和评估模型
                auc_results = []
                
                for i in range(1, repeat + 1):
                    print(f'######################## the {i}-th repeat ########################')
                    if seed != -1:
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        print('Set seed to %d.' % seed)
                    
                    # 准备数据集
                    data_dim = normal_data.shape[1]
                    train_data, train_lab, test_data, test_lab = split_train_test(normal_data, abnormal_data, n_train_data=n_train_data)
                    dataset = TabularDataset(
                                        train_data=train_data,
                                        train_lab=train_lab,
                                        test_data=test_data,
                                        test_lab=test_lab)
            
                    # 构建模型
                    assert n_d > 0 and n_g > 0
                    IGAN = GANsTrainer(
                                        optimizer_name=optimizer_name,
                                        lr_d=lr_d,
                                        lr_g=lr_g,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        device=device,
                                        latent_dim=current_latent_dim,
                                        n_d=n_d,
                                        n_g=n_g)
        
                    IGAN.build_networks(generator=Generator(data_dim, current_latent_dim), discriminator=Discriminator(current_latent_dim, 1))
                    
                    # 训练模型
                    if train:
                        metrics = IGAN.train(dataset=dataset)
                        auc_results.append(metrics.get('auroc', 0))
                
                # 计算平均AUC
                avg_auc = np.mean(auc_results) if auc_results else 0
                results_ms_igan.append([avg_auc, 0, 0, 0])  # 占位符，可以根据需要添加更多指标
                print(f"Results for {file}: AUC_IGAN={avg_auc:.4f}")
                
            except Exception as e:
                print(f"!!! Error processing {file}: {e}")
                traceback.print_exc()
                results_ms_igan.append([np.nan]*4)
        
        # 保存结果
        if dataset_names:
            output_filepath = os.path.join(results_dir, "result_MS_IGAN_AllDatasets.xlsx")
            output_df = pd.DataFrame(data=results_ms_igan, 
                                    columns=['AUC_IGAN', 'Metric2', 'Metric3', 'Metric4'], 
                                    index=dataset_names)
            output_df = output_df.sort_index()
            output_df.to_excel(output_filepath)
            print(f"\nResults saved to {output_filepath}")
    else:
        # 处理单个数据集（原始逻辑）
        if train: # train mode
            for i in range(1, repeat + 1):
                print(f'######################## the {i}-th repeat ########################')
                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.backends.cudnn.deterministic = True
                    print('Set seed to %d.' % seed)
                
                # load dataset =============================
                normal_data, abnormal_data = load_data(dataset=data_dir)
                data_dim = normal_data.shape[1]
                train_data, train_lab, test_data, test_lab = split_train_test(normal_data, abnormal_data, n_train_data=n_train_data)
                dataset = TabularDataset(
                                        train_data=train_data,
                                        train_lab=train_lab,
                                        test_data=test_data,
                                        test_lab=test_lab)
        
                # model =====================================
                assert n_d > 0 and n_g > 0
                IGAN = GANsTrainer(
                                    optimizer_name=optimizer_name,
                                    lr_d=lr_d,
                                    lr_g=lr_g,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    device=device,
                                    latent_dim=latent_dim,
                                    n_d=n_d,
                                    n_g=n_g)

                IGAN.build_networks(generator=Generator(data_dim, latent_dim), discriminator=Discriminator(latent_dim, 1))
                
                # train ======================================
                IGAN.train(dataset=dataset)

    print(f'===============================  End  =============================== ')


if __name__  == '__main__':
    main()




