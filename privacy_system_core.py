import torch
import numpy as np
import os
import time
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from scipy.spatial import cKDTree
from network.RandLANet import Network as RandLANet
import urllib.request


def download_weight(weight_path="pretrain_model/checkpoint.tar"):
    if os.path.exists(weight_path):
        print(f"权重文件已存在: {weight_path}")
        return True
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    url = "https://github.com/atori777/myproject/releases/download/V1.0/checkpoint.tar"
    try:
        urllib.request.urlretrieve(url, weight_path)
        print(f"权重下载完成: {weight_path}")
        return True
    except Exception as e:
        print(f"权重下载失败: {e}")
        return False


class Config:
    def __init__(self):
        self.input_dim = 3
        self.num_classes = 19
        self.num_layers = 4
        self.device = 'cpu'
        self.batch_size = 1
        self.d_out = [16, 64, 128, 256]


class PointPrivacyEngine:
    def __init__(self, num_classes=19, device=None, pretrain_path="pretrain_model/checkpoint.tar"):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = Config()
        config.num_classes = num_classes
        config.device = self.device
        self.model = RandLANet(config)
        self.model = self.model.to(self.device)
        download_weight(pretrain_path)
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载预训练权重: {pretrain_path}")
        else:
            print(f"未找到预训练权重: {pretrain_path}")
        self.model.eval()
        self.key = AESGCM.generate_key(bit_length=128)
        self.aesgcm = AESGCM(self.key)
        self.privacy_labels = [10, 11, 13, 18]

    def _up_sample_labels(self, original_xyz, sampled_xyz, sampled_labels):
        tree = cKDTree(sampled_xyz)
        _, indices = tree.query(original_xyz, k=1)
        return sampled_labels[indices]

    def protect_frame(self, bin_path):
        raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        original_xyz = raw_data[:, :3]
        if len(original_xyz) > 50000:
            idx = np.random.choice(len(original_xyz), 50000, replace=False)
            input_points = original_xyz[idx, :]
        else:
            input_points = original_xyz
        xyz_tensor = torch.from_numpy(input_points).float().unsqueeze(0).to(self.device)
        from utils.data_process import DataProcessing
        neigh_idx = DataProcessing.knn_search(input_points, input_points, k=16)
        neigh_idx_tensor = torch.from_numpy(neigh_idx).long().to(self.device)
        input_dict = {
            'features': xyz_tensor.transpose(1, 2),
            'xyz': xyz_tensor,
            'neigh_idx': neigh_idx_tensor
        }
        with torch.no_grad():
            t_start = time.time()
            output = self.model(input_dict)
            logits = output['logits']
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            t_inference = time.time() - t_start
        full_labels = self._up_sample_labels(original_xyz, input_points, preds)
        mask = np.isin(full_labels, self.privacy_labels)
        if np.sum(mask) == 0:
            dist = np.linalg.norm(original_xyz, axis=1)
            mask = (dist > 3) & (dist < 20) & (original_xyz[:, 2] > -1.6) & (original_xyz[:, 2] < 0.2)
        target_points = original_xyz[mask]
        background_points = original_xyz[~mask]
        nonce = os.urandom(12)
        if len(target_points) > 0:
            target_bytes = target_points.astype(np.float32).tobytes()
            ciphertext = self.aesgcm.encrypt(nonce, target_bytes, None)
        else:
            ciphertext = None
        return {
            "inference_time": t_inference * 1000,
            "nonce": nonce,
            "ciphertext": ciphertext,
            "background": background_points,
            "target_count": len(target_points),
            "target_shape": target_points.shape if len(target_points) > 0 else None,
            "mask": mask
        }

    def authorize_recovery(self, packet):
        if packet["ciphertext"] is None:
            return packet["background"]
        decrypted_data = self.aesgcm.decrypt(packet["nonce"], packet["ciphertext"], None)
        recovered_points = np.frombuffer(decrypted_data, dtype=np.float32).reshape(packet["target_shape"])
        return np.vstack((packet["background"], recovered_points))


if __name__ == "__main__":
    engine = PointPrivacyEngine()
    print("系统已就绪，感知引擎与安全引擎已完成逻辑对齐。")

