import torch
import numpy as np
import os
import time
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from scipy.spatial import cKDTree
from network.RandLANet import Network as RandLANet

#privacy_system_core.py

import urllib.request


def download_weight(weight_path="pretrain_model/checkpoint.tar"):
    """从 GitHub Release 下载预训练权重"""
    if os.path.exists(weight_path):
        print(f"✅ 权重文件已存在: {weight_path}")
        return True

    # 创建文件夹
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    # 下载链接
    url = "https://github.com/atori777/myproject/releases/download/V1.0/checkpoint.tar"

    print(f"正在下载预训练权重...")
    try:
        urllib.request.urlretrieve(url, weight_path)
        print(f"✅ 权重下载完成: {weight_path}")
        return True
    except Exception as e:
        print(f"❌ 权重下载失败: {e}")
        return False
# 简单的配置类，用于初始化模型
class Config:
    def __init__(self):
        self.input_dim = 3  # 输入维度 (x,y,z)
        self.num_classes = 19  # SemanticKITTI 类别数
        self.num_layers = 4  # 网络层数
        self.device = 'cpu'  # 设备
        self.batch_size = 1  # 推理时 batch_size=1

        # 各层的输出维度（编码器）
        self.d_out = [16, 64, 128, 256]


class PointPrivacyEngine:
    def __init__(self, num_classes=19, device=None, pretrain_path="pretrain_model/checkpoint.tar"):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建配置
        config = Config()
        config.num_classes = num_classes
        config.device = self.device

        # 创建模型
        self.model = RandLANet(config)
        self.model = self.model.to(self.device)

        download_weight(pretrain_path) #下载权重
        # 加载预训练权重
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ 成功加载预训练权重: {pretrain_path}")
        else:
            print(f"⚠️ 未找到预训练权重: {pretrain_path}")

        self.model.eval()

        self.key = AESGCM.generate_key(bit_length=128)
        self.aesgcm = AESGCM(self.key)
        self.privacy_labels = [10, 11, 13, 18]  # SemanticKITTI: Car, Bicycle, Bus, Truck

    def _up_sample_labels(self, original_xyz, sampled_xyz, sampled_labels):
        tree = cKDTree(sampled_xyz)
        _, indices = tree.query(original_xyz, k=1)
        return sampled_labels[indices]

    def protect_frame(self, bin_path):
        # A. 数据加载
        raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        original_xyz = raw_data[:, :3]

        # B. 随机采样
        indices = np.random.choice(len(original_xyz), min(65536, len(original_xyz)), replace=False)
        input_points = original_xyz[indices, :]
        input_tensor = torch.from_numpy(input_points).unsqueeze(0).to(self.device)  # (1, N, 3)

      with torch.no_grad():
            t_start = time.time()

            # 1. 模拟 RandLA-Net 的 4 层下采样 (必须严格符合模型维度)
            current_xyz = input_points
            all_xyz = []
            all_neigh_idx = []
            
            # 每一层的下采样倍率 (第一层不采样)
            sub_sample_ratio = [1, 4, 4, 4] 
            
            for i in range(4):
                # 计算目标点数
                target_n = len(current_xyz) // sub_sample_ratio[i]
                if target_n < 512: target_n = 512 # 保护底线
                
                # 随机采样
                idx = np.random.choice(len(current_xyz), target_n, replace=False)
                current_xyz = current_xyz[idx]
                
                # 计算当前层的 KNN (k=16)
                tree = cKDTree(current_xyz)
                _, neigh_idx = tree.query(current_xyz, k=16)
                
                # --- 关键修改：转换为 Tensor 并强制增加 Batch 维度 (1, N, ...) ---
                # 使用 .float() 确保是 float32，.long() 确保索引是整数
                xyz_tensor = torch.from_numpy(current_xyz).float().unsqueeze(0).to(self.device)
                neigh_tensor = torch.from_numpy(neigh_idx).long().unsqueeze(0).to(self.device)
                
                all_xyz.append(xyz_tensor)
                all_neigh_idx.append(neigh_tensor)

            # 2. 组装输入字典 (必须和 RandLANet.py 的 forward 预期一致)
            input_dict = {
                'xyz': all_xyz,                # 4层 [1, N_i, 3] 的列表
                'neigh_idx': all_neigh_idx,    # 4层 [1, N_i, 16] 的列表
                'features': all_xyz[0].transpose(1, 2) # 特征输入应为 [1, 3, N]
            }

            # 3. 推理
            logits = self.model(input_dict)
            # 得到结果 [1, classes, N] -> [N]
            sampled_preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            t_inference = time.time() - t_start

        # D. 如果模型失败，回退到规则方法
        if np.sum(mask) == 0:
            dist = np.linalg.norm(original_xyz, axis=1)
            mask = (dist > 3) & (dist < 20) & (original_xyz[:, 2] > -1.6) & (original_xyz[:, 2] < 0.2)

        target_points = original_xyz[mask]
        background_points = original_xyz[~mask]

        # E. AES-GCM加密
        nonce = os.urandom(12)
        if len(target_points) > 0:
            ciphertext = self.aesgcm.encrypt(nonce, target_points.tobytes(), None)
        else:
            ciphertext = None

        return {
            "inference_time": t_inference,
            "nonce": nonce,
            "ciphertext": ciphertext,
            "background": background_points,
            "target_count": len(target_points),
            "target_shape": target_points.shape if len(target_points) > 0 else None,
            "mask": mask
        }
    def authorize_recovery(self, packet):
        if packet["ciphertext"] is None: return packet["background"]
        decrypted_data = self.aesgcm.decrypt(packet["nonce"], packet["ciphertext"], None)
        recovered_points = np.frombuffer(decrypted_data, dtype=np.float32).reshape(packet["target_shape"])
        return np.vstack((packet["background"], recovered_points))


if __name__ == "__main__":
    engine = PointPrivacyEngine()
    print("✅ 系统已就绪，感知引擎与安全引擎已完成逻辑对齐。")
