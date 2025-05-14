# IGAN_files/dataset/tabular.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset # 确保 DataLoader 和 Dataset 已导入

# --- 从 IGAN 项目的 utils 导入 ---
# 根据你的项目结构，这里的导入可能需要调整
# 假设 utils.py 位于 IGAN_files 目录下，而 tabular.py 在 IGAN_files/dataset/ 目录下
try:
    from ..utils import obj_save, ROOT_DIR as IGAN_ROOT_DIR, generate_filename, new_dir
except ImportError:
    # 后备导入，如果直接运行此文件或结构不同
    # 这要求 IGAN_files 目录在 PYTHONPATH 中，或者脚本从能找到 utils 的地方运行
    print("IGAN TabularDataset: Warning: Could not import IGAN utils with relative path. "
          "Attempting direct import (ensure IGAN_files is in PYTHONPATH or utils.py is accessible).")
    # 尝试直接导入 (假设 utils.py 在某个可访问的路径)
    # 或者，如果 utils.py 与 tabular.py 在同一文件夹 (不符合 IGAN 原始结构)
    # from utils import obj_save, ROOT_DIR as IGAN_ROOT_DIR, generate_filename, new_dir
    #
    # 为了使代码能运行，这里做一个简化的后备，如果严格的导入失败
    def obj_save(path, obj): print(f"Mock obj_save: Skipping save to {path}")
    def new_dir(father_dir, mk_dir):
        path = os.path.join(father_dir, mk_dir or "temp_results_igan_tabular")
        # if not os.path.exists(path): os.makedirs(path) # 在封装器中创建目录，这里不主动创建
        return path
    def generate_filename(suffix, *args, **kwargs): return f"mean_std_temp{suffix}"
    IGAN_ROOT_DIR = "."
    print("IGAN TabularDataset: Using mock utility functions due to import error. Mean/std file saving might be affected.")


# --- 从 IGAN 项目的 base_dataset 导入 ---
try:
    from .base_dataset import TorchvisionDataset # . 表示当前目录
except ImportError:
    # 后备导入
    from base_dataset import TorchvisionDataset


class CustomDataset(Dataset):
    def __init__(self, data, labels): # data 应该是 numpy array
        self.data = data.astype(np.float32) # 确保数据类型
        self.labels = torch.from_numpy(labels.astype(np.float32)).type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _data = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
        return _data, (self.labels[idx]), idx


class TabularDataset(TorchvisionDataset):
      def __init__(self, train_data, train_lab, test_data, test_lab, mode='train',
                   results_save_dir=None): # 用于控制可选地保存 mean/std 到文件
        super().__init__('') # IGAN 原始的 root 是空字符串
        self.name = 'tabular'
        self.mean = None  # 初始化为 None
        self.std = None   # 初始化为 None

        if mode == 'train':
            print('IGAN TabularDataset: Initializing in TRAIN mode.')
            if train_data is None or train_data.shape[0] == 0:
                print("  Warning: train_data is empty or None in TabularDataset init (train mode). Mean/Std will be None.")
                # 创建空的 CustomDataset 以避免后续错误
                self.train_set = CustomDataset(np.array([], dtype=np.float32).reshape(0, test_data.shape[1] if test_data is not None and test_data.shape[0] > 0 else 0), np.array([]))
                self.test_set = CustomDataset(np.array([], dtype=np.float32).reshape(0, test_data.shape[1] if test_data is not None and test_data.shape[0] > 0 else 0), np.array([]))
                return # 直接返回

            # --- 基于传入的 train_data 计算并存储 mean 和 std ---
            self.mean = np.mean(train_data, axis=0, dtype=np.float32)
            self.std = np.std(train_data, axis=0, dtype=np.float32)
            # ---

            std_adjusted = np.where(self.std < 1e-6, 1e-6, self.std) # 避免除以零

            train_data_standardized = (train_data - self.mean) / std_adjusted
            # 测试数据也应该使用训练数据的 mean/std 进行标准化
            if test_data is not None and test_data.shape[0] > 0:
                test_data_standardized = (test_data - self.mean) / std_adjusted
            else: # 如果 test_data 为空或 None
                test_data_standardized = np.array([], dtype=np.float32).reshape(0, train_data.shape[1] if train_data.shape[0] > 0 else 0)
                if test_lab is None : test_lab = np.array([])


            # print('============ data set (IGAN TabularDataset internal after standardization) ================')
            # print(f'train data shape: {train_data_standardized.shape}')
            # print(f'test data shape: {test_data_standardized.shape}')
            # print('=======================================================================================')

            self.train_set = CustomDataset(train_data_standardized, train_lab)
            self.test_set = CustomDataset(test_data_standardized, test_lab)

            # 可选：原始 IGAN 保存 mean/std 到文件的逻辑
            if results_save_dir and self.mean is not None and self.std is not None:
                if not os.path.exists(results_save_dir):
                    try:
                        os.makedirs(results_save_dir)
                        print(f"  IGAN TabularDataset: Created directory {results_save_dir}")
                    except Exception as e_dir:
                        print(f"  IGAN TabularDataset: Error creating directory {results_save_dir}: {e_dir}. Skipping save.")
                        results_save_dir = None # 阻止保存

                if results_save_dir: #再次检查，如果创建失败
                    mean_std_file_name = generate_filename('.pkl', *['mean', 'std'], timestamp=False)
                    save_path_file = os.path.join(results_save_dir, mean_std_file_name)
                    try:
                        obj_save(save_path_file, {'mean': self.mean, 'std': self.std})
                        print(f"  IGAN TabularDataset: Saved mean/std to {save_path_file}")
                    except Exception as e_save:
                        print(f"  IGAN TabularDataset: Error saving mean/std to {save_path_file}: {e_save}")
            elif results_save_dir:
                 print(f"  IGAN TabularDataset: Mean/std not computed, skipping save to {results_save_dir}.")


        elif mode == 'test':
            print('IGAN TabularDataset: Initializing in TEST mode.')
            # 在我们的封装器 IGAN_Scorer_for_MGBOD 中，
            # predict 方法会使用 fit 阶段保存的 mean/std 来手动标准化测试数据。
            # 所以这里的 test_set 可以直接使用传入的 test_data (未标准化的)。
            # 训练器 GANsTrainer 的 test 方法也期望其 dataset 的 test_loader 返回标准化数据。
            # 因此，如果 TabularDataset 在 test mode 下要被 GANsTrainer.test() 使用，
            # 它需要一种方式来获取或被传入正确的 mean 和 std。
            # 在我们当前的 MS-IGAN 框架中，IGAN_Scorer 的 predict 是直接调用的，
            # 不经过 GANsTrainer.test()，所以这里的 TabularDataset (test mode) 相对简单。
            if test_data is None or test_lab is None:
                print("  Warning: test_data or test_lab is None in TabularDataset (test mode).")
                self.test_set = CustomDataset(np.array([]).reshape(0,train_data.shape[1] if train_data is not None and train_data.shape[0]>0 else 0), np.array([]))
            else:
                self.test_set = CustomDataset(test_data, test_lab) # 假设外部已处理标准化
        else:
            raise Exception(f'Unknown mode [{mode}]!')

      # loaders 方法保持不变，但确保其 drop_last 参数被正确使用
      def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0,
                  drop_last_train=True, drop_last_test=False) -> ( # 返回类型提示
            DataLoader, DataLoader):
          train_loader = None
          if self.train_set is not None and len(self.train_set) > 0 :
              train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                    num_workers=num_workers, drop_last=drop_last_train)

          test_loader = None # 初始化为 None
          if self.test_set is not None and len(self.test_set) > 0:
              test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                   num_workers=num_workers, drop_last=drop_last_test)
          return train_loader, test_loader