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

    print(f"正在下载预训练权重...")
    try:
        urllib.request.urlretrieve(url, weight_path)
        print(f"权重下载完成: {weight_path}")
        return True
    except Exception as e:
        print(f"权重下载失败: {e}")
        return False


def knn_search(support_pts, query_pts, k):
    if support_pts.ndim == 2 and query_pts.ndim == 2:
        tree = cKDTree(support_pts)
        _, neighbor_idx = tree.query(query_pts, k=k)
        return neighbor_idx.astype(np.int32)
    B = support_pts.shape[0]
    all_idx = []
    for b in range(B):
        tree = cKDTree(support_pts[b])
        _, neighbor_idx = tree.query(query_pts[b], k=k)
        all_idx.append(neighbor_idx.astype(np.int32))
    return np.stack(all_idx, axis=0)


class Config:
    def __init__(self):
        self.input_dim = 3
        self.num_classes = 19
        self.num_layers = 4
        self.k_n = 16
        self.num_points = 4096 * 11
        self.sub_sampling_ratio = [4, 4, 4, 4]
        self.d_out = [16, 64, 128, 256]
        self.device = 'cpu'
        self.batch_size = 1


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
        self.config = config

    def _up_sample_labels(self, original_xyz, sampled_xyz, sampled_labels):
        tree = cKDTree(sampled_xyz)
        _, indices = tree.query(original_xyz, k=1)
        return sampled_labels[indices]

    def _prepare_inputs(self, points_xyz):
        B = 1
        N = points_xyz.shape[0]

        xyz = []
        neigh_idx = []
        sub_idx = []
        interp_idx = []

        current_pc = points_xyz.copy()
        for i in range(self.config.num_layers):
            neighbor = knn_search(current_pc, current_pc, self.config.k_n)
            neigh_idx.append(torch.from_numpy(neighbor).long().unsqueeze(0))

            xyz.append(torch.from_numpy(current_pc).float().unsqueeze(0))

            step = self.config.sub_sampling_ratio[i]
            sub_idx_arr = np.arange(0, current_pc.shape[0], step, dtype=np.int32)
            n_sub = len(sub_idx_arr)

            sub_idx_tensor = torch.zeros((1, n_sub, 1), dtype=torch.long)
            sub_idx_tensor[0, :, 0] = torch.from_numpy(sub_idx_arr)
            sub_idx.append(sub_idx_tensor)

            sub_points = current_pc[sub_idx_arr, :]

            up_idx = np.zeros((B, current_pc.shape[0], 1), dtype=np.int32)
            tree = cKDTree(sub_points)
            _, nearest = tree.query(current_pc, k=1)
            up_idx[0, :, 0] = nearest
            interp_idx.append(torch.from_numpy(up_idx).long())

            current_pc = sub_points

        features = torch.from_numpy(points_xyz).float().unsqueeze(0).transpose(1, 2)
        labels = torch.zeros((B, N), dtype=torch.long)

        inputs = {
            'xyz': xyz,
            'neigh_idx': neigh_idx,
            'sub_idx': sub_idx,
            'interp_idx': interp_idx,
            'features': features,
            'labels': labels
        }

        return inputs

    def protect_frame(self, bin_path):
        raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        original_xyz = raw_data[:, :3]

        num_points = self.config.num_points
        if len(original_xyz) > num_points:
            choices = np.random.choice(len(original_xyz), num_points, replace=False)
            points_xyz = original_xyz[choices]
        else:
            deficit = num_points - len(original_xyz)
            zeros = np.zeros((deficit, 3), dtype=original_xyz.dtype)
            points_xyz = np.vstack([original_xyz, zeros])

        input_dict = self._prepare_inputs(points_xyz)

        for key in input_dict:
            if isinstance(input_dict[key], list):
                input_dict[key] = [x.to(self.device) for x in input_dict[key]]
            else:
                input_dict[key] = input_dict[key].to(self.device)

        with torch.no_grad():
            t_start = time.time()
            output = self.model(input_dict)
            logits = output['logits']
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            t_inference = time.time() - t_start

        valid_mask = (points_xyz[:, 0] != 0) | (points_xyz[:, 1] != 0) | (points_xyz[:, 2] != 0)
        valid_preds = preds[valid_mask]
        valid_xyz = original_xyz

        full_labels = self._up_sample_labels(valid_xyz, points_xyz[valid_mask], valid_preds)
        mask = np.isin(full_labels, self.privacy_labels)

        if np.sum(mask) == 0:
            dist = np.linalg.norm(valid_xyz, axis=1)
            mask = (dist > 3) & (dist < 20) & (valid_xyz[:, 2] > -1.6) & (valid_xyz[:, 2] < 0.2)

        target_points = valid_xyz[mask]
        background_points = valid_xyz[~mask]

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
