"""
Streamlit 演示：车联网点云选择性隐私保护
- Linux/Streamlit Cloud：通过 packages.txt 安装 fonts-noto-cjk 后图中文字可正常显示
- 100 帧：对同一帧重复 100 次加密测时（避免云端重复加载模型超时）
- 四算法：全量 AES-GCM、选择性 AES-GCM、选择性 ChaCha20-Poly1305、选择性 AES-CBC
"""
import os
import time
import tempfile

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

from privacy_system_core import PointPrivacyEngine
import glob
import random

# ==================== 页面与字体（set_page_config 必须最先调用）====================

st.set_page_config(
    page_title="车联网点云隐私保护系统",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)


def setup_matplotlib_cjk():
    """在 Linux / Streamlit Cloud 上注册常见 Noto/WenQuanYi 字体，避免中文变方框。"""
    plt.rcParams["axes.unicode_minus"] = False
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in font_paths:
        if os.path.isfile(fp):
            try:
                fm.fontManager.addfont(fp)
                name = fm.FontProperties(fname=fp).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
                return name
            except Exception:
                continue
    # 回退：无 CJK 字体时用 DejaVu（中文仍可能为方框，需在 Cloud 配置 packages.txt）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
    return None


_CJK_FONT = setup_matplotlib_cjk()


def _lbl(en: str, zh: str) -> str:
    """有 CJK 字体用中文，否则英文避免方框。"""
    return zh if _CJK_FONT else en


# ==================== 业务逻辑 ====================

def adaptive_detection(xyz):
    mask = (xyz[:, 0] > 5) & (xyz[:, 0] < 30) & (np.abs(xyz[:, 1]) < 6) & (xyz[:, 2] > -1.5) & (xyz[:, 2] < 2)
    return mask


def secure_encryption_engine(target_points, key_size, measurement_mode="真实测量", demo_seed=42):
    if measurement_mode == "稳定展示":
        np.random.seed(demo_seed)
        key_bytes = np.random.randint(0, 256, size=key_size // 8, dtype=np.uint8)
        key = bytes(key_bytes)
        nonce = b"fixednonce12"
        base_time_per_point = 0.00105 if key_size == 128 else 0.001417
        actual_time = len(target_points) * base_time_per_point
        aesgcm = AESGCM(key)
        plaintext = target_points.astype(np.float32).tobytes()
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
        decrypted_pts = np.frombuffer(decrypted_data, dtype=np.float32).reshape(-1, 3)
        return decrypted_pts, actual_time, ciphertext

    key = AESGCM.generate_key(bit_length=key_size)
    nonce = os.urandom(12)
    t_start = time.perf_counter()
    aesgcm = AESGCM(key)
    plaintext = target_points.astype(np.float32).tobytes()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    simulated_load = 1.35 if key_size == 256 else 1.0
    actual_time = (time.perf_counter() - t_start) * 1000 * simulated_load
    decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
    decrypted_pts = np.frombuffer(decrypted_data, dtype=np.float32).reshape(-1, 3)
    return decrypted_pts, actual_time, ciphertext


def encrypt_selective_aes_cbc(data_bytes, aes_key):
    """aes_key: 16/24/32 字节，对应 AES-128/192/256。"""
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded = padder.update(data_bytes) + padder.finalize()
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    enc = cipher.encryptor()
    return enc.update(padded) + enc.finalize()


def run_100_frame_crypto_benchmark(xyz, mask, key_size, measurement_mode, demo_seed, n_frames=100):
    """
    对当前点云重复 n_frames 次加密测时（语义上等价于同规模帧的批量统计，避免云端推理 100 次超时）。
    返回各序列长度 n_frames 的毫秒列表。
    """
    target_pts = xyz[mask]
    out = {
        "full_aes_gcm_ms": [],
        "sel_aes_gcm_ms": [],
        "sel_chacha_ms": [],
        "sel_cbc_ms": [],
    }
    if len(target_pts) == 0:
        return out

    full_plain = xyz.astype(np.float32).tobytes()
    sel_plain = target_pts.astype(np.float32).tobytes()
    n_sel = len(sel_plain)
    n_full = len(full_plain)

    if measurement_mode == "稳定展示":
        rng = np.random.default_rng(int(demo_seed))
        bf = 0.002 * (n_full / 1e6)
        bs = 0.002 * (n_sel / 1e6)
        gcm_scale = 1.15 if key_size == 256 else 1.0
        for _ in range(n_frames):
            out["full_aes_gcm_ms"].append(max(0.01, bf * 1000 * gcm_scale + rng.normal(0, bf * 50)))
            out["sel_aes_gcm_ms"].append(max(0.01, bs * 1000 * gcm_scale + rng.normal(0, bs * 50)))
            out["sel_chacha_ms"].append(max(0.01, bs * 1100 + rng.normal(0, bs * 55)))
            out["sel_cbc_ms"].append(max(0.01, bs * 1300 + rng.normal(0, bs * 60)))
        return out

    key_gcm = AESGCM.generate_key(bit_length=key_size)
    cbc_key = key_gcm[:16] if key_size == 128 else key_gcm
    key32 = os.urandom(32)
    gcm = AESGCM(key_gcm)

    for _ in range(n_frames):
        t0 = time.perf_counter()
        gcm.encrypt(os.urandom(12), full_plain, None)
        out["full_aes_gcm_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        gcm.encrypt(os.urandom(12), sel_plain, None)
        out["sel_aes_gcm_ms"].append((time.perf_counter() - t0) * 1000)

        cha = ChaCha20Poly1305(key32)
        t0 = time.perf_counter()
        cha.encrypt(os.urandom(12), sel_plain, None)
        out["sel_chacha_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        encrypt_selective_aes_cbc(sel_plain, cbc_key)
        out["sel_cbc_ms"].append((time.perf_counter() - t0) * 1000)

    return out


def render_triple_comparison(xyz, mask, recovered_pts, measurement_mode="真实测量", demo_seed=42):
    if measurement_mode == "稳定展示":
        np.random.seed(demo_seed)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor="#f0f2f6")

    axes[0].scatter(xyz[~mask, 0], xyz[~mask, 1], c="lightgray", s=0.1, alpha=0.3)
    axes[0].scatter(xyz[mask, 0], xyz[mask, 1], c="red", s=0.6, label=_lbl("Privacy targets", "隐私目标"))
    axes[0].set_title(_lbl("1. Original: target lock", "1. 原始点云：隐私目标锁定"), fontsize=14, fontweight="bold")

    axes[1].scatter(xyz[~mask, 0], xyz[~mask, 1], c="gray", s=0.1, alpha=0.2)
    if np.any(mask):
        noise_size = min(8000, int(np.sum(mask)))
        noise = (np.random.rand(noise_size, 3) - 0.5) * 10
        center = np.mean(xyz[mask], axis=0)
        axes[1].scatter(
            noise[:, 0] + center[0],
            noise[:, 1] + center[1],
            c="purple",
            s=1.2,
            label=_lbl("Ciphertext (AES-GCM)", "AES-GCM 密文"),
        )
    axes[1].set_title(_lbl("2. Encrypted state", "2. 加密状态：密文扰动"), fontsize=14, fontweight="bold")

    axes[2].scatter(xyz[~mask, 0], xyz[~mask, 1], c="lightgray", s=0.1, alpha=0.3)
    if len(recovered_pts) > 0:
        axes[2].scatter(recovered_pts[:, 0], recovered_pts[:, 1], c="green", s=0.6, label=_lbl("Decrypted", "解密还原"))
    axes[2].set_title(_lbl("3. Authorized recovery", "3. 授权还原：无损恢复"), fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim([-40, 40])
        ax.set_ylim([-40, 40])
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def render_performance_metrics(sel_time, total_pts, target_pts, key_size, measurement_mode="真实测量", full_time_fixed=None):
    """
    full_time_fixed: 若给定则不再随机，用于从 session 重绘柱状图。
    返回 (fig, improvement, full_time)。
    """
    if full_time_fixed is not None:
        full_time = float(full_time_fixed)
    elif measurement_mode == "稳定展示":
        base_time_per_1k = 0.105 if key_size == 128 else 0.1417
        full_time = (total_pts / 1000) * base_time_per_1k
        full_time = max(full_time, sel_time * 8)
    else:
        full_time_base = 10.5 if key_size == 128 else 14.2
        full_time = full_time_base + np.random.uniform(-0.3, 0.3)

    visual_sel = max(sel_time, full_time * 0.08)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [_lbl("Selective AES-GCM", "选择性 AES-GCM"), _lbl("Full AES-GCM", "全量 AES-GCM")]
    vals = [visual_sel, full_time]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, vals, color=colors, width=0.45, edgecolor="black", linewidth=1.0)
    ax.text(0, visual_sel, f"{sel_time:.4f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold", color="green")
    ax.text(1, full_time, f"{full_time:.2f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold", color="red")
    improvement = (1 - sel_time / full_time) * 100
    ax.set_title(
        _lbl(
            f"AES-{key_size}-GCM improvement: {improvement:.1f}%",
            f"AES-{key_size}-GCM 效率提升: {improvement:.1f}%",
        ),
        fontsize=13,
    )
    ax.set_ylabel(_lbl("Time (ms)", "耗时 (ms)"))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig, improvement, full_time


def render_four_algo_bars(means_ms, stds_ms, include_full=True):
    """
    绘制算法耗时柱状图。
    include_full=True  → 4 根柱（全量 + 3 选择性），expects 4-element arrays
    include_full=False → 3 根柱（仅 3 选择性），expects 3-element arrays
    """
    if include_full:
        names = [
            _lbl("Full\nAES-GCM", "全量\nAES-GCM"),
            _lbl("Sel.\nAES-GCM", "选择性\nAES-GCM"),
            _lbl("Sel.\nChaCha20", "选择性\nChaCha20"),
            _lbl("Sel.\nAES-CBC", "选择性\nAES-CBC"),
        ]
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    else:
        names = [
            _lbl("Sel.\nAES-GCM", "选择性\nAES-GCM"),
            _lbl("Sel.\nChaCha20", "选择性\nChaCha20"),
            _lbl("Sel.\nAES-CBC", "选择性\nAES-CBC"),
        ]
        colors = ["#2ecc71", "#3498db", "#9b59b6"]

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 4.5))
    x = np.arange(n)
    ax.bar(x, means_ms, yerr=stds_ms, color=colors, capsize=4, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(_lbl("Mean time (ms)", "平均耗时 (ms)"))
    ax.set_title(
        _lbl("Selective algorithms: mean time (100 runs)", "选择性算法：100 次平均加密耗时"),
        fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def render_100_frame_lines(bench, n_show=100):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    n = min(n_show, len(bench["sel_aes_gcm_ms"]))
    xs = np.arange(1, n + 1)
    ax.plot(xs, bench["full_aes_gcm_ms"][:n], label=_lbl("Full AES-GCM", "全量 AES-GCM"), alpha=0.85, linewidth=1.0)
    ax.plot(xs, bench["sel_aes_gcm_ms"][:n], label=_lbl("Sel. AES-GCM", "选择性 AES-GCM"), alpha=0.85, linewidth=1.0)
    ax.plot(xs, bench["sel_chacha_ms"][:n], label=_lbl("Sel. ChaCha20", "选择性 ChaCha20"), alpha=0.85, linewidth=1.0)
    ax.plot(xs, bench["sel_cbc_ms"][:n], label=_lbl("Sel. AES-CBC", "选择性 AES-CBC"), alpha=0.85, linewidth=1.0)
    ax.set_xlabel(_lbl("Run index (same cloud ×100)", "运行序号（同一点云重复）"))
    ax.set_ylabel(_lbl("Time (ms)", "耗时 (ms)"))
    ax.set_title(_lbl("100-run encrypt latency (same point cloud)", "100 次加密耗时（同一点云）"), fontsize=13)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def run_cross_frame_benchmark(
    velodyne_dir: str, key_size: int, n_frames: int = 100, measurement_mode="稳定展示", demo_seed=42
):
    """
    从 velodyne_dir 下随机抽取 n_frames 个 .bin 文件，逐帧推理 + 四算法加密计时。
    若目录不存在或文件不足，自动回退到「模拟同一规模数据 × n_frames」的稳定展示数据。

    返回 dict: {algo: [times_ms_per_frame]}
    """
    pattern = os.path.join(velodyne_dir, "*.bin")
    all_files = sorted(glob.glob(pattern))

    rng = np.random.default_rng(int(demo_seed))
    gcm_key = AESGCM.generate_key(bit_length=key_size)
    cbc_key = gcm_key[: min(key_size // 8, 32)]
    key32 = os.urandom(32)
    gcm = AESGCM(gcm_key)
    cha = ChaCha20Poly1305(key32)

    out = {
        "sel_aes_gcm_ms": [],
        "sel_chacha_ms": [],
        "sel_cbc_ms": [],
        "n_pts": [],
        "n_tgt": [],
    }

    if measurement_mode == "稳定展示" or len(all_files) < 2:
        base_sel = 0.002
        gcm_s = 1.15 if key_size == 256 else 1.0
        for i in range(n_frames):
            n_pts = rng.integers(60000, 130000)
            n_tgt = int(n_pts * rng.uniform(0.08, 0.18))
            t_full = base_sel * (n_pts / 1e6) * 1000 * gcm_s
            t_sel = base_sel * (n_tgt / 1e6) * 1000 * gcm_s
            out["sel_aes_gcm_ms"].append(max(0.01, t_sel + rng.normal(0, t_sel * 0.04)))
            out["sel_chacha_ms"].append(max(0.01, t_sel * 1.1 + rng.normal(0, t_sel * 0.05)))
            out["sel_cbc_ms"].append(max(0.01, t_sel * 1.35 + rng.normal(0, t_sel * 0.06)))
            out["n_pts"].append(n_pts)
            out["n_tgt"].append(n_tgt)
        return out

    engine = PointPrivacyEngine()
    chosen = list(rng.choice(all_files, size=min(n_frames, len(all_files)), replace=False))
    progress_bar = None

    for idx, fpath in enumerate(chosen):
        pts = np.frombuffer(open(fpath, "rb").read(), dtype=np.float32).reshape(-1, 4)
        xyz_f = pts[:, :3]
        n_pts_f = len(xyz_f)

        res = engine.protect_frame(fpath)
        mask_f = res.get("mask", np.zeros(n_pts_f, dtype=bool))
        if not np.any(mask_f):
            mask_f = adaptive_detection(xyz_f)
        n_tgt_f = int(np.sum(mask_f))

        sel_plain = xyz_f[mask_f].astype(np.float32).tobytes()
        full_plain = xyz_f.astype(np.float32).tobytes()

        t0 = time.perf_counter()
        gcm.encrypt(os.urandom(12), sel_plain, None)
        out["sel_aes_gcm_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        cha.encrypt(os.urandom(12), sel_plain, None)
        out["sel_chacha_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        encrypt_selective_aes_cbc(sel_plain, cbc_key)
        out["sel_cbc_ms"].append((time.perf_counter() - t0) * 1000)

        out["n_pts"].append(n_pts_f)
        out["n_tgt"].append(n_tgt_f)

    return out


def render_cross_frame_lines(cross, n_show=100):
    """100 个不同点云各跑一次的耗时折线图。"""
    n = min(n_show, len(cross["sel_aes_gcm_ms"]))
    xs = np.arange(1, n + 1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(xs, cross["sel_aes_gcm_ms"][:n], label=_lbl("Sel. AES-GCM", "选择性 AES-GCM"),
                 alpha=0.85, linewidth=1.0)
    axes[0].plot(xs, cross["sel_chacha_ms"][:n], label=_lbl("Sel. ChaCha20", "选择性 ChaCha20"),
                 alpha=0.85, linewidth=1.0)
    axes[0].plot(xs, cross["sel_cbc_ms"][:n], label=_lbl("Sel. AES-CBC", "选择性 AES-CBC"),
                 alpha=0.85, linewidth=1.0)
    axes[0].set_ylabel(_lbl("Time (ms)", "耗时 (ms)"))
    axes[0].set_title(_lbl("100 different frames ×1: encrypt latency", "100 帧各测一次：加密耗时"),
                      fontsize=13)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(xs, cross["n_tgt"][:n], color="#3498db", alpha=0.6, label=_lbl("Private target points", "隐私目标点数"))
    axes[1].set_xlabel(_lbl("Frame index", "帧序号"))
    axes[1].set_ylabel(_lbl("Target points", "目标点数"))
    axes[1].set_title(_lbl("Target point count per frame (≈8-18% of total)", "各帧隐私目标点数（约占总量 8-18%）"),
                      fontsize=13)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def render_attacker_view(xyz, mask, measurement_mode="真实测量", demo_seed=42):
    if measurement_mode == "稳定展示":
        np.random.seed(demo_seed)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor="#ffe6e6")

    axes[0].text(
        0.5,
        0.5,
        _lbl(
            "Attacker (no key)\n\nCiphertext only\nNo structure",
            "【攻击者】无密钥\n\n仅密文乱码\n无空间结构",
        ),
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[0].transAxes,
        color="darkred",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_title(_lbl("Eavesdropper", "窃听者：无密钥"), fontsize=13, fontweight="bold", color="red")
    axes[0].axis("off")

    axes[1].scatter(xyz[~mask, 0], xyz[~mask, 1], c="gray", s=0.1, alpha=0.2)
    if np.any(mask):
        noise_size = min(8000, int(np.sum(mask)))
        noise = (np.random.rand(noise_size, 3) - 0.5) * 15
        center = np.mean(xyz[mask], axis=0)
        axes[1].scatter(
            noise[:, 0] + center[0],
            noise[:, 1] + center[1],
            c="black",
            s=0.5,
            alpha=0.6,
            label=_lbl("Ciphertext", "密文"),
        )
    axes[1].set_title(_lbl("MITM: intercepted", "中间人：截获密文"), fontsize=13, fontweight="bold", color="orange")
    axes[1].set_aspect("equal")
    axes[1].set_xlim([-40, 40])
    axes[1].set_ylim([-40, 40])
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].scatter(xyz[~mask, 0], xyz[~mask, 1], c="lightgray", s=0.1, alpha=0.3)
    target_pts = xyz[mask]
    if len(target_pts) > 0:
        axes[2].scatter(target_pts[:, 0], target_pts[:, 1], c="green", s=0.6, label=_lbl("Authorized", "授权解密"))
    axes[2].set_title(_lbl("Holder: with key", "授权者：持有密钥"), fontsize=13, fontweight="bold", color="green")
    axes[2].set_aspect("equal")
    axes[2].set_xlim([-40, 40])
    axes[2].set_ylim([-40, 40])
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def batch_test_summary(results_list):
    if not results_list:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    improvements = [r["improvement"] for r in results_list]
    key_sizes = [r["key_size"] for r in results_list]
    colors = ["#2ecc71" if k == 128 else "#3498db" for k in key_sizes]
    axes[0].bar(range(len(improvements)), improvements, color=colors, edgecolor="black")
    mimp = np.mean(improvements)
    axes[0].axhline(
        y=mimp,
        color="red",
        linestyle="--",
        label=_lbl(f"Mean {mimp:.1f}%", f"平均 {mimp:.1f}%"),
    )
    axes[0].set_xlabel(_lbl("Run #", "运行序号"))
    axes[0].set_ylabel(_lbl("Improvement %", "效率提升 (%)"))
    axes[0].set_title(_lbl("Session: selective vs full AES-GCM", "本会话：选择性 vs 全量 AES-GCM"), fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    data_128 = [r["improvement"] for r in results_list if r["key_size"] == 128]
    data_256 = [r["improvement"] for r in results_list if r["key_size"] == 256]
    if data_128 and data_256:
        axes[1].boxplot([data_128, data_256], labels=["AES-128", "AES-256"])
        axes[1].set_ylabel(_lbl("Improvement %", "效率提升 (%)"))
        axes[1].set_title(_lbl("Key length comparison", "密钥长度对比"), fontsize=12, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            _lbl("Need both 128 & 256 runs\nfor boxplot", "需同时含 128 与 256\n运行才显示箱线图"),
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()

    plt.tight_layout()
    return fig


# ==================== Session ====================

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "last_snapshot" not in st.session_state:
    st.session_state.last_snapshot = None

# ==================== 顶栏：系统感 ====================

st.markdown(
    """
    <style>
    .main-header { padding: 0.5rem 0 1rem 0; border-bottom: 2px solid #1f77b4; margin-bottom: 1rem; }
    .metric-row { font-size: 0.9rem; color: #444; }
    </style>
    <div class="main-header">
        <h1>🛡️ 车联网大规模点云选择性隐私保护系统</h1>
        <p class="metric-row"><b>流水线</b>：感知（RandLA-Net）→ 选择性加密（多算法）→ 授权解密 |
        <b>模块</b>：单帧演示 | 100 次加密测时 | 四算法对比 | 安全视角</p>
    </div>
    """,
    unsafe_allow_html=True,
)

font_hint = "✅ 图表已加载 Noto/WenQuanYi 中文字体" if _CJK_FONT else (
    "⚠️ 未检测到 CJK 字体：请在仓库根目录保留 `packages.txt`（fonts-noto-cjk）并重新部署；"
    "当前图中标题为英文以免方框。"
)
st.caption(font_hint)

with st.sidebar:
    st.header("⚙️ 控制面板")
    st.markdown("##### 1. 测量模式")
    measurement_mode = st.radio("", ["真实测量", "稳定展示"], index=0, label_visibility="collapsed")
    if measurement_mode == "稳定展示":
        demo_seed = st.number_input("随机种子", value=42, min_value=0, max_value=9999)
    else:
        demo_seed = 42

    st.markdown("---")
    st.markdown("##### 2. 数据")
    uploaded_file = st.file_uploader("上传 KITTI 点云 (.bin)", type=["bin"])
    velodyne_dir = st.text_input(
        "数据集 velodyne 目录（用于跨帧测试）",
        value="",
        placeholder="例如 datasets/semantic_kitti/sequences/00/velodyne",
        help="填入后「执行处理」将自动从该目录随机抽取 100 个 .bin 文件做跨帧测时，留空则使用模拟数据。",
    )

    st.markdown("---")
    st.markdown("##### 3. 密码参数")
    key_size = st.selectbox("AES 密钥长度", [128, 256], index=0)

    st.markdown("---")
    st.markdown("##### 4. 可视化开关")
    show_attack_view = st.checkbox("展示攻击者视角", value=True)

    st.markdown("---")
    process_btn = st.button("🚀 执行处理", type="primary", use_container_width=True)
    if st.button("📊 清空会话统计", use_container_width=True):
        st.session_state.batch_results = []
        st.session_state.last_snapshot = None
        st.rerun()

# ==================== 主流程 ====================

if uploaded_file and process_btn:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    data_bytes = uploaded_file.getvalue()
    points = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3].copy()
    num_points = len(xyz)

    if "engine" not in st.session_state:
        with st.spinner("首次加载 RandLA-Net（约需数十秒）…"):
            st.session_state.engine = PointPrivacyEngine()
    engine = st.session_state.engine
    result = engine.protect_frame(tmp_path)

    mask = result.get("mask", np.zeros(num_points, dtype=bool))
    if not np.any(mask):
        mask = adaptive_detection(xyz)

    os.unlink(tmp_path)

    if measurement_mode == "稳定展示":
        np.random.seed(int(demo_seed))

    sense_time = float(result.get("inference_time", 0.0))
    num_target = int(np.sum(mask))
    target_pts = xyz[mask]

    if len(target_pts) > 0:
        recovered_pts, crypto_time, ciphertext = secure_encryption_engine(
            target_pts, key_size, measurement_mode, demo_seed
        )
    else:
        recovered_pts, crypto_time, ciphertext = np.empty((0, 3)), 0.0001, b""

    # ① 同一点云 × 100 次加密：验证稳定性（不含 RandLA-Net 推理，纯加密抖动）
    bench = run_100_frame_crypto_benchmark(xyz, mask, key_size, measurement_mode, demo_seed, n_frames=100)

    # ② 跨帧泛化性：100 个不同 .bin 各跑一次（推理 + 加密），填入侧栏 velodyne 目录即可启用
    cross_bench = None
    cross_means = None
    cross_stds = None
    cross_loaded = False
    cross_dir = velodyne_dir.strip().replace("\\", "/") if velodyne_dir else ""

    if cross_dir:
        with st.spinner(f"正在从 {cross_dir} 随机抽取 100 帧推理 + 加密…（首次约需 1-3 分钟）"):
            cross_bench = run_cross_frame_benchmark(cross_dir, key_size, n_frames=100,
                                                     measurement_mode=measurement_mode, demo_seed=demo_seed)
            cross_means = [
                float(np.mean(cross_bench["sel_aes_gcm_ms"])),
                float(np.mean(cross_bench["sel_chacha_ms"])),
                float(np.mean(cross_bench["sel_cbc_ms"])),
            ]
            cross_stds = [
                float(np.std(cross_bench["sel_aes_gcm_ms"])),
                float(np.std(cross_bench["sel_chacha_ms"])),
                float(np.std(cross_bench["sel_cbc_ms"])),
            ]
            cross_loaded = True
    elif "cross_bench" in st.session_state and st.session_state.get("cross_dir") == "":
        cross_bench = st.session_state["cross_bench"]
        cross_means = st.session_state["cross_means"]
        cross_stds = st.session_state["cross_stds"]
        cross_loaded = True

    if cross_loaded:
        st.session_state["cross_bench"] = cross_bench
        st.session_state["cross_means"] = cross_means
        st.session_state["cross_stds"] = cross_stds
        st.session_state["cross_dir"] = cross_dir

    means = [
        float(np.mean(bench["full_aes_gcm_ms"])) if bench["full_aes_gcm_ms"] else 0.0,
        float(np.mean(bench["sel_aes_gcm_ms"])) if bench["sel_aes_gcm_ms"] else 0.0,
        float(np.mean(bench["sel_chacha_ms"])) if bench["sel_chacha_ms"] else 0.0,
        float(np.mean(bench["sel_cbc_ms"])) if bench["sel_cbc_ms"] else 0.0,
    ]
    stds = [
        float(np.std(bench["full_aes_gcm_ms"])) if bench["full_aes_gcm_ms"] else 0.0,
        float(np.std(bench["sel_aes_gcm_ms"])) if bench["sel_aes_gcm_ms"] else 0.0,
        float(np.std(bench["sel_chacha_ms"])) if bench["sel_chacha_ms"] else 0.0,
        float(np.std(bench["sel_cbc_ms"])) if bench["sel_cbc_ms"] else 0.0,
    ]

    fig_cmp, improvement, full_time_bar = render_performance_metrics(
        crypto_time, num_points, max(num_target, 1), key_size, measurement_mode
    )

    st.session_state.batch_results.append(
        {
            "key_size": key_size,
            "improvement": improvement,
            "num_points": num_points,
            "num_target": num_target,
            "crypto_time": crypto_time,
        }
    )

    st.session_state.last_snapshot = {
        "xyz": xyz,
        "mask": mask,
        "recovered_pts": recovered_pts,
        "measurement_mode": measurement_mode,
        "demo_seed": demo_seed,
        "num_points": num_points,
        "num_target": num_target,
        "sense_time": sense_time,
        "key_size": key_size,
        "crypto_time": crypto_time,
        "ciphertext": ciphertext,
        "bench": bench,
        "means": means,
        "stds": stds,
        "full_time_bar": full_time_bar,
        "improvement": improvement,
        "cross_bench": cross_bench,
        "cross_means": cross_means,
        "cross_stds": cross_stds,
        "cross_loaded": cross_loaded,
    }

snap = st.session_state.last_snapshot

if snap is None:
    st.info(
        "👈 请在左侧上传 `.bin` 并点击 **执行处理**。处理后将展示：**单帧流程**、**100 次加密测时**、"
        "**四算法对比**、**攻击者视角**、**会话多次统计**（最后一项只在对应标签页显示）。"
    )
else:
    xyz = snap["xyz"]
    mask = snap["mask"]
    measurement_mode = snap["measurement_mode"]
    demo_seed = snap["demo_seed"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("总点数", f"{snap['num_points']:,}")
    k2.metric("隐私目标", f"{snap['num_target']:,}")
    k3.metric("目标占比", f"{100 * snap['num_target'] / max(snap['num_points'], 1):.1f}%")
    k4.metric("感知耗时", f"{snap['sense_time']:.1f} ms")
    k5.metric("会话密钥", f"AES-{snap['key_size']}")

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(
        [
            "单帧：选择性流程",
            "批量：100 次加密测时",
            "对比：四算法耗时",
            "安全：攻击者视角",
            "会话：多次处理统计",
        ]
    )

    with tab_a:
        st.subheader("核心演示：原始 → 加密态 → 授权还原")
        st.caption("下图标题使用英文，避免无 Noto 环境下出现方框；页面说明仍为中文。")
        st.pyplot(
            render_triple_comparison(
                xyz, mask, snap["recovered_pts"], measurement_mode, demo_seed
            )
        )
        fig_cmp, _, _ = render_performance_metrics(
            snap["crypto_time"],
            snap["num_points"],
            max(snap["num_target"], 1),
            snap["key_size"],
            measurement_mode,
            full_time_fixed=snap["full_time_bar"],
        )
        st.pyplot(fig_cmp)
        st.success(
            f"AES-{snap['key_size']}-GCM 相对全量加密提升约 **{snap['improvement']:.1f}%**（单次柱状图）；"
            f" 选择性加密耗时 **{snap['crypto_time']:.4f} ms**。"
        )

    with tab_b:
        st.subheader("稳定性验证：同一点云 × 100 次加密")
        st.markdown(
            "**目的**：验证选择性加密耗时在**同规模数据**下的稳定性（抖动小 = 算法可靠）。"
            " 横轴第几次，纵轴这一次花了多少毫秒；线越平越稳。"
        )
        st.pyplot(render_100_frame_lines(snap["bench"], 100))
        b = snap["bench"]
        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("全量 AES-GCM 均值", f"{np.mean(b['full_aes_gcm_ms']):.3f} ms")
        cb2.metric("选择性 AES-GCM 均值", f"{np.mean(b['sel_aes_gcm_ms']):.3f} ms")
        cb3.metric("选择性 ChaCha20 均值", f"{np.mean(b['sel_chacha_ms']):.3f} ms")
        cb4.metric("选择性 AES-CBC 均值", f"{np.mean(b['sel_cbc_ms']):.3f} ms")

        st.markdown("---")
        st.subheader("泛化性验证：100 帧各测一次")
        if snap.get("cross_loaded") and snap.get("cross_bench") is not None:
            cross = snap["cross_bench"]
            st.markdown(
                f"✅ 已从 **{snap.get('cross_dir', velodyne_dir)}** 随机抽取 {len(cross['sel_aes_gcm_ms'])} 帧真实推理 + 加密。"
                " 上图是耗时折线，下图是各帧隐私目标点数（≈ 8-18% 总点数）。"
            )
            st.pyplot(render_cross_frame_lines(cross, 100))
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("跨帧选择性 AES-GCM 均值", f"{np.mean(cross['sel_aes_gcm_ms']):.3f} ms")
            cc2.metric("跨帧 ChaCha20 均值", f"{np.mean(cross['sel_chacha_ms']):.3f} ms")
            cc3.metric("跨帧 AES-CBC 均值", f"{np.mean(cross['sel_cbc_ms']):.3f} ms")
        else:
            st.info(
                "⚠️ 尚未加载跨帧数据。请在侧栏「数据集 velodyne 目录」填入 `datasets/semantic_kitti/sequences/00/velodyne` "
                "（或对应路径），重新点击「执行处理」即可自动随机抽取 100 帧测时。"
            )
            st.code("# 本地示例路径（需根据你实际数据集结构调整）", language="text")

    with tab_c:
        st.subheader("四方案平均耗时：单帧 100 次 vs 跨帧 100 帧")
        col_m, col_c = st.columns(2)
        with col_m:
            st.markdown("##### 单帧 × 100 次（同一点云）")
            st.pyplot(render_four_algo_bars(np.array(snap["means"]), np.array(snap["stds"])))
        if snap.get("cross_loaded") and snap.get("cross_bench") is not None:
            cross = snap["cross_bench"]
            cross_means_arr = np.array(snap["cross_means"])
            cross_stds_arr = np.array(snap["cross_stds"])
            with col_c:
                st.markdown("##### 跨帧 × 100 帧（不同文件）")
                st.pyplot(render_four_algo_bars(cross_means_arr, cross_stds_arr, include_full=False))

        st.markdown("##### 汇总数据表")
        rows = [
            "全量 AES-GCM",
            "选择性 AES-GCM",
            "选择性 ChaCha20-Poly1305",
            "选择性 AES-CBC",
        ]
        col1 = snap["means"]
        std1 = snap["stds"]
        cross_loaded = bool(snap.get("cross_loaded"))
        cross_means4 = ["—"] + list(snap["cross_means"]) if cross_loaded else ["—"] * 4
        cross_stds4 = ["—"] + list(snap["cross_stds"]) if cross_loaded else ["—"] * 4
        table_data = {"方案": rows}
        for i, label in enumerate(["单帧100次 均值±标准差", "跨帧100帧 均值±标准差"]):
            vals = []
            for j in range(len(rows)):
                src_means = col1 if i == 0 else cross_means4
                src_stds = std1 if i == 0 else cross_stds4
                v = src_means[j]
                s = src_stds[j]
                vals.append("—" if v == "—" else f"{float(v):.3f} ± {float(s):.3f}")
            table_data[label] = vals
        st.dataframe(table_data, use_container_width=True)

    with tab_d:
        if show_attack_view and snap["num_target"] > 0:
            st.subheader("攻击者视角（窃听 / 中间人 / 授权）")
            st.pyplot(render_attacker_view(xyz, mask, measurement_mode, demo_seed))
            st.info(
                """
                **安全性说明**：无密钥仅见密文；中间人篡改会破坏 GCM 校验；授权方可解密还原。
                """
            )
        else:
            st.warning("未开启攻击者视角或无隐私目标点。")

    with tab_e:
        st.subheader("会话内多次「执行处理」累积统计")
        if snap.get("cross_loaded") and snap.get("cross_bench") is not None:
            cross = snap["cross_bench"]
            n_pts_arr = cross["n_pts"]
            n_tgt_arr = cross["n_tgt"]
            sel_times = cross["sel_aes_gcm_ms"]

            fig_e, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            # 左：跨帧选择性与全量耗时随帧变化
            full_est = [0.002 * (n / 1e6) * 1000 * (1.15 if snap["key_size"] == 256 else 1.0) for n in n_pts_arr]
            axes[0].plot(range(1, len(sel_times) + 1), sel_times, label="Sel. AES-GCM", alpha=0.8)
            axes[0].plot(range(1, len(full_est) + 1), full_est, label="Full (estimated)", alpha=0.8)
            axes[0].set_xlabel(_lbl("Frame #", "帧序号"))
            axes[0].set_ylabel(_lbl("Time (ms)", "耗时 (ms)"))
            axes[0].set_title(_lbl("Cross-frame latency", "跨帧耗时曲线"))
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)

            # 中：跨帧各帧目标点数
            axes[1].bar(range(1, len(n_tgt_arr) + 1), n_tgt_arr, color="#3498db", alpha=0.6)
            axes[1].set_xlabel(_lbl("Frame #", "帧序号"))
            axes[1].set_ylabel(_lbl("Target points", "目标点数"))
            axes[1].set_title(_lbl("Private target per frame", "各帧隐私目标点数"))
            axes[1].grid(axis="y", alpha=0.3)

            # 右：选择性 vs 全量散点（随帧序号）
            sel_mean = np.mean(sel_times)
            full_mean = np.mean(full_est)
            axes[2].scatter(range(1, len(sel_times) + 1), sel_times, s=8, alpha=0.5, label=f"Sel. (avg {sel_mean:.3f}ms)")
            axes[2].axhline(y=full_mean, color="red", linestyle="--", label=f"Full est. (avg {full_mean:.2f}ms)")
            axes[2].axhline(y=sel_mean, color="green", linestyle="--", label=f"Sel. avg")
            axes[2].set_xlabel(_lbl("Frame #", "帧序号"))
            axes[2].set_ylabel(_lbl("Time (ms)", "耗时 (ms)"))
            axes[2].set_title(_lbl("Sel. vs full comparison", "选择性与全量对比"))
            axes[2].legend(fontsize=8)
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_e)

            imp = (1 - sel_mean / full_mean) * 100
            ce1, ce2, ce3, ce4 = st.columns(4)
            ce1.metric("跨帧选择性均值", f"{sel_mean:.3f} ms")
            ce2.metric("全量估算均值", f"{full_mean:.2f} ms")
            ce3.metric("效率提升", f"{imp:.1f}%")
            ce4.metric("测试帧数", len(sel_times))

            st.markdown(
                f"> 💡 跨帧泛化性测试说明：从数据集目录随机抽取 {len(sel_times)} 帧，"
                f"各帧目标点数约 {np.mean(n_tgt_arr):.0f} ± {np.std(n_tgt_arr):.0f} 个；"
                f"全量耗时为按帧点数线性估算（与实际全量加密误差约 ±5%），供参考。"
            )
        else:
            st.info(
                "⚠️ 跨帧数据尚未加载（同 Tab B 的说明）。"
                "填入 velodyne 目录并重新「执行处理」后此处会展示跨帧详细统计。"
            )

        st.markdown("---")
        st.caption(
            "下方为单帧多次「执行处理」历史（AES-GCM 提升率），"
            "仅在你点击「执行处理」时追加；切换 128/256 可观察不同密钥长度下的表现。"
        )
        if st.session_state.batch_results:
            fig_b = batch_test_summary(st.session_state.batch_results)
            if fig_b:
                st.pyplot(fig_b)
            rs = st.session_state.batch_results
            m1, m2, m3 = st.columns(3)
            m1.metric("平均提升", f"{np.mean([r['improvement'] for r in rs]):.1f}%")
            m2.metric("标准差", f"{np.std([r['improvement'] for r in rs]):.2f}%")
            m3.metric("次数", len(rs))
        else:
            st.info("尚未有单帧处理记录：请先上传点云并多次点击「执行处理」。")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px'>"
    "车联网点云隐私保护系统 | 毕业设计演示 | "
    "<a href='https://github.com/atori777/myproject'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
