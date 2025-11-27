#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
安全稳定的 Hugging Face 下载脚本
支持：
  - 国内镜像 https://hf-mirror.com
  - resume_download 断点续传
  - 自动重试与日志输出
  - 可直接下载到大容量磁盘 local_dir
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"      # 国内镜像
os.environ["HF_HUB_DISABLE_XETFS"] = "true"         # 禁用 XetFS
os.environ["DISABLE_XETFS"] = "1"
from huggingface_hub import snapshot_download, HfApi

import time
import logging


# ======================================================
# 1️⃣ 环境配置 —— 一定要在 import huggingface_hub 之前执行
# ======================================================


# 可选：若需要断点缓存，可启用自己的 cache 路径
# os.environ["HF_HOME"] = "/data/.cache/huggingface"

# ======================================================
# 2️⃣ 基础日志
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hf_downloader")

# ======================================================
# 3️⃣ 检查端点是否生效
# ======================================================
api = HfApi()
logger.info(f"当前端点: {api.endpoint}")
if "hf-mirror" not in api.endpoint:
    logger.warning("⚠️ 镜像未生效，请确认 os.environ 设置在 import 前。")

# ======================================================
# 4️⃣ 下载函数（自动重试）
# ======================================================


def download_with_retry(repo_id, repo_type, local_dir, allow_patterns=None, max_retries=10):
    retry = 0
    while True:
        try:
            logger.info(f"开始下载 {repo_id} (尝试 {retry + 1}/{max_retries})")
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,   # ✅ 支持断点续传
                force_download=False,
                max_workers=4,
            )
            logger.info(f"✅ 下载完成，保存路径：{path}")
            break
        except Exception as e:
            retry += 1
            if retry >= max_retries:
                logger.error(f"❌ 达到最大重试次数 ({max_retries})，下载失败。错误：{e}")
                break
            wait = min(300, retry * 30)
            logger.warning(f"⚠️ 下载出错 ({e})，{wait}s 后重试...")
            time.sleep(wait)


# ======================================================
# 5️⃣ 调用
# ======================================================
if __name__ == "__main__":
    download_with_retry(
        repo_id="lerobot-raw/droid_100_raw",
        repo_type="dataset",
        local_dir="/data/nf/datasets/droid100"
    )
