"""
Randomly sample TFRecord episodes from multiple datasets, export image frames,
and create GIFs plus instruction text dumps for quick visualization.

Usage:
1. 在 `dataset.yaml` 中填写 `output_root` 和各数据集的 `path`。
2. (可选) 根据需要增删 image/instruction 键。
3. 运行 `python Get.py`，每个数据集随机导出两个 episode。
"""

from __future__ import annotations

import io
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import tensorflow as tf
import yaml
from PIL import Image


# === Config ====================================================================

DatasetConfig = Dict[str, Any]
DatasetConfigMap = Dict[str, DatasetConfig]
CONFIG_PATH = Path(__file__).with_name("dataset2.yaml")

# Sampling knobs.
TFRECORDS_PER_DATASET = 5
EPISODES_PER_DATASET = 4
RANDOM_SEED = 7878

# Export options.
FPS = 10
# set to an int to cap frames, or None to keep all frames in GIF.
GIF_MAX_FRAMES = None
IMAGE_EXTENSION = "jpg"


# === Helpers ===================================================================


def ensure_tf_uses_cpu() -> None:
    """Avoid TensorFlow grabbing GPU memory when this script runs."""
    print("1111")
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def list_tfrecord_files(dataset_path: Path) -> List[Path]:
    files = sorted(
        file for file in dataset_path.glob("*.tfrecord*") if file.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {dataset_path}")
    return files


def select_files(files: Sequence[Path], k: int) -> List[Path]:
    if len(files) <= k:
        return list(files)
    return random.sample(list(files), k)


def build_feature_description(image_keys: Sequence[str], instruction_keys: Sequence[str]) -> Dict[str, tf.io.VarLenFeature]:
    desc: Dict[str, tf.io.VarLenFeature] = {}
    for key in set(image_keys + instruction_keys):
        desc[key] = tf.io.VarLenFeature(tf.string)
    return desc


def sample_episodes_from_file(
    tfrecord_path: Path,
    feature_description: Dict[str, tf.io.VarLenFeature],
    requested: int,
) -> List[Tuple[int, Dict[str, tf.Tensor]]]:
    """Reservoir sample a fixed number of episodes from a single TFRecord shard."""
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    reservoir: List[Tuple[int, Dict[str, tf.Tensor]]] = []
    for idx, raw_example in enumerate(dataset):
        parsed = tf.io.parse_single_example(raw_example, feature_description)
        if len(reservoir) < requested:
            reservoir.append((idx, parsed))
        else:
            swap_idx = random.randint(0, idx)
            if swap_idx < requested:
                reservoir[swap_idx] = (idx, parsed)
    return reservoir


def safe_name(key: str) -> str:
    return key.replace("/", "__")


def decode_frames(varlen_tensor: tf.sparse.SparseTensor) -> List[Image.Image]:
    if varlen_tensor is None:
        return []
    frames: List[Image.Image] = []
    for raw in varlen_tensor.values.numpy():
        try:
            frame = Image.open(io.BytesIO(raw)).convert("RGB")
            frames.append(frame)
        except Exception as exc:
            print(f"    ⚠️ Failed to decode frame: {exc}")
    return frames


def save_frames_and_gif(
    frames: List[Image.Image],
    target_dir: Path,
    basename: str,
    *,
    frame_prefix: str | None = None,
    gif_name_override: str | None = None,
) -> None:
    if not frames:
        return
    frame_dir = target_dir / f"{basename}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_name = f"{frame_prefix}{idx:04d}" if frame_prefix else f"{idx:04d}"
        frame.save(frame_dir / f"{frame_name}.{IMAGE_EXTENSION}", quality=95)
    gif_stem = gif_name_override or basename
    gif_path = target_dir / f"{gif_stem}.gif"
    trimmed = frames if GIF_MAX_FRAMES in (
        None, 0) else frames[:GIF_MAX_FRAMES]
    duration_ms = max(int(1000 / max(FPS, 1)), 1)
    trimmed[0].save(
        gif_path,
        save_all=True,
        append_images=trimmed[1:],
        format="GIF",
        duration=duration_ms,
        loop=0,
    )


def extract_instructions(varlen_tensor: tf.sparse.SparseTensor) -> List[str]:
    if varlen_tensor is None:
        return []
    texts: List[str] = []
    for raw in varlen_tensor.values.numpy():
        try:
            texts.append(raw.decode("utf-8", errors="ignore"))
        except Exception:
            # numpy.bytes_ also supports decode directly.
            texts.append(str(raw))
    return texts


def write_instruction_file(
    instructions: Dict[str, List[str]],
    target_dir: Path,
) -> None:
    if not instructions:
        return
    lines: List[str] = []
    for key, texts in instructions.items():
        lines.append(f"{key}:")
        if texts:
            for idx, text in enumerate(texts):
                lines.append(f"  [{idx}] {text}")
        else:
            lines.append("  <empty>")
        lines.append("")
    (target_dir / "instructions.txt").write_text("\n".join(lines).strip() +
                                                 "\n", encoding="utf-8")


def slugify_filename(text: str, fallback: str = "instruction", max_len: int = 80) -> str:
    cleaned = re.sub(r"\s+", "_", text.strip())
    cleaned = re.sub(r"[^\w\-]+", "_", cleaned)
    cleaned = cleaned.strip("_") or fallback
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned or fallback


def extract_primary_instruction(instructions: Dict[str, List[str]]) -> str | None:
    for texts in instructions.values():
        for text in texts:
            stripped = text.strip()
            if stripped:
                return stripped
    return None


def is_libero_dataset(name: str) -> bool:
    return "libero" in name.lower()


def load_yaml_config() -> Tuple[str, DatasetConfigMap]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("dataset.yaml 需要包含 datasets 字段")
    output_root = str(payload.get("output_root", "") or "").strip()
    return output_root, datasets


def process_dataset(
    dataset_name: str,
    config: DatasetConfig,
    run_root: Path,
) -> None:
    dataset_path_str = str(config.get("path", "")).strip()
    if not dataset_path_str:
        print(f"➡️  Skipping {dataset_name}: dataset path not set.")
        return
    dataset_path = Path(dataset_path_str).expanduser()
    image_keys = list(config.get("image_keys", []))
    instruction_keys = list(config.get("instruction_keys", []))
    if not image_keys and not instruction_keys:
        print(f"➡️  Skipping {dataset_name}: no keys defined.")
        return

    print(f"\n=== Processing {dataset_name} ===")
    print(f"  Dataset path: {dataset_path}")
    all_tfrecords = list_tfrecord_files(dataset_path)
    tfrecord_pick = 1 if is_libero_dataset(
        dataset_name) else TFRECORDS_PER_DATASET
    tfrecord_files = select_files(all_tfrecords, tfrecord_pick)
    print(
        f"  Total TFRecords found: {len(all_tfrecords)}, selected: {len(tfrecord_files)}")
    for idx, tf_path in enumerate(tfrecord_files, start=1):
        print(f"    [{idx}] {tf_path.name}")
    feature_description = build_feature_description(
        image_keys, instruction_keys)
    sampled: List[Tuple[int, Dict[str, tf.Tensor], str, int]] = []
    episode_counter = 0
    for shard_idx, tfrecord_path in enumerate(tfrecord_files, start=1):
        print(
            f"  🔹 Reading TFRecord {shard_idx}/{len(tfrecord_files)}: {tfrecord_path.name}")
        per_file_samples = sample_episodes_from_file(
            tfrecord_path, feature_description, EPISODES_PER_DATASET)
        print(
            f"    → Sampled {len(per_file_samples)} / {EPISODES_PER_DATASET} episodes from {tfrecord_path.name}")
        if not per_file_samples:
            print(f"    ⚠️ {tfrecord_path.name} 没有采到可用 episode。")
            continue
        for local_idx, example in per_file_samples:
            sampled.append((episode_counter, example,
                           tfrecord_path.name, local_idx))
            episode_counter += 1
    if not sampled:
        print("  ⚠️ 所选 TFRecord 中没有采到 episode。")
        return

    dataset_output = run_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    random.shuffle(sampled)
    for episode_idx, example, shard_name, local_idx in sampled:
        episode_dir = dataset_output / f"episode_{episode_idx:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  • Saving episode {episode_idx} (shard={shard_name}, local={local_idx}) -> {episode_dir}")

        instruction_dump: Dict[str, List[str]] = {}
        for key in instruction_keys:
            tensor = example.get(key)
            texts = extract_instructions(tensor)
            instruction_dump[key] = texts
        write_instruction_file(instruction_dump, episode_dir)
        primary_instruction = extract_primary_instruction(instruction_dump)
        gif_basename = slugify_filename(
            primary_instruction) if primary_instruction else f"episode_{episode_idx:05d}"

        for key in image_keys:
            tensor = example.get(key)
            frames = decode_frames(tensor)
            if not frames:
                print(f"    ⚠️ No frames found for key: {key}")
                continue
            frame_prefix = "frame"
            if len(image_keys) > 1:
                gif_name_override = slugify_filename(
                    f"{gif_basename}_{safe_name(key)}")
            else:
                gif_name_override = gif_basename
            save_frames_and_gif(
                frames,
                episode_dir,
                safe_name(key),
                frame_prefix=frame_prefix,
                gif_name_override=gif_name_override,
            )


def main() -> None:
    output_root_str, dataset_configs = load_yaml_config()
    if not output_root_str:
        raise ValueError("请在 dataset.yaml 中设置 output_root。")
    random.seed(RANDOM_SEED)
    ensure_tf_uses_cpu()
    output_root = Path(output_root_str).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / f"seed{RANDOM_SEED}"
    run_root.mkdir(parents=True, exist_ok=True)
    for dataset_name, config in dataset_configs.items():
        if not isinstance(config, dict):
            print(f"➡️  Skipping {dataset_name}: 配置不是字典。")
            continue
        process_dataset(dataset_name, config, run_root)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
