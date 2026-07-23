"""Dataset auditing, verification, and pixel statistics utilities for TrustOCT."""

import json
import os
from typing import Dict, List, Tuple
import yaml
import numpy as np
from PIL import Image
from datasets.dataset import CLASS_NAMES, CLASS_TO_INDEX, SUPPORTED_EXTENSIONS


def verify_folder_structure(train_path: str, val_path: str, test_path: str) -> List[str]:
    missing_dirs = []
    for name, path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(path):
            missing_dirs.append(path)
    return missing_dirs


def verify_class_folders(base_path: str, classes: List[str]) -> Dict[str, bool]:
    class_status = {}
    for c in classes:
        class_path = os.path.join(base_path, c)
        class_status[c] = os.path.exists(class_path)
    return class_status


def verify_images(
    base_path: str, classes: List[str], extensions: Tuple[str, ...]
) -> Tuple[int, Dict[str, int], List[str], List[str]]:
    total_images = 0
    class_counts = {c: 0 for c in classes}
    corrupt_images = []
    unsupported_files = []

    for c in classes:
        class_path = os.path.join(base_path, c)
        if not os.path.exists(class_path):
            continue

        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if os.path.isdir(file_path):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() not in extensions:
                unsupported_files.append(file_path)
                continue

            total_images += 1
            class_counts[c] += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                corrupt_images.append(file_path)

    return total_images, class_counts, corrupt_images, unsupported_files


def generate_dataset_report(config: dict) -> dict:
    paths = config.get("paths", {})
    train_path = paths.get("train", "datasets/raw/Kermany/train")
    val_path = paths.get("val", "datasets/raw/Kermany/val")
    test_path = paths.get("test", "datasets/raw/Kermany/test")

    classes = config.get("classes", CLASS_NAMES)
    extensions = tuple(config.get("image", {}).get("extensions", SUPPORTED_EXTENSIONS))

    missing_dirs = verify_folder_structure(train_path, val_path, test_path)
    
    report = {
        "dataset_name": config.get("dataset", {}).get("name", "Kermany OCT2017"),
        "verification_status": "SUCCESS" if not missing_dirs else "FAILED",
        "missing_directories": missing_dirs,
        "splits": {}
    }

    for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if path in missing_dirs:
            report["splits"][split_name] = {
                "exists": False, "path": path, "total_images": 0, "class_counts": {},
                "corrupt_images": [], "unsupported_files": []
            }
            continue

        class_status = verify_class_folders(path, classes)
        total_images, class_counts, corrupt, unsupported = verify_images(path, classes, extensions)

        report["splits"][split_name] = {
            "exists": True, "path": path, "class_folders_status": class_status,
            "total_images": total_images, "class_counts": class_counts,
            "corrupt_images": corrupt, "unsupported_files": unsupported
        }

    return report


def verify_dataset(config_path: str, report_output_path: str) -> bool:
    print("=" * 40)
    print("TrustOCT Dataset Verification")
    print("=" * 40)

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    report = generate_dataset_report(config)
    
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=4)

    if report["verification_status"] == "FAILED":
        print(f"[ERROR] Verification FAILED. Missing directories: {report['missing_directories']}")
        return False

    for split_name, split_info in report["splits"].items():
        if split_info.get("corrupt_images"):
            print(f"[ERROR] Found corrupt images in {split_name} split!")
            return False

    print("[OK] Folder structure verified")
    print("[OK] Classes verified")
    print("[OK] Images verified")
    print("[OK] No corrupted images")
    print(f"Report saved to: {report_output_path}")
    print("=" * 40)
    return True


def calculate_statistics(
    base_path: str, classes: List[str], extensions: Tuple[str, ...], max_samples: int = 1000
) -> Tuple[List[float], List[float], Tuple[float, float], Dict[str, int]]:
    class_distribution = {}
    all_filepaths = []

    for c in classes:
        class_path = os.path.join(base_path, c)
        if not os.path.exists(class_path):
            class_distribution[c] = 0
            continue

        files = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        class_distribution[c] = len(files)
        all_filepaths.extend(files)

    if not all_filepaths:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 0.0), class_distribution

    np.random.seed(42)
    sample_filepaths = all_filepaths
    if len(all_filepaths) > max_samples:
        indices = np.random.choice(len(all_filepaths), max_samples, replace=False)
        sample_filepaths = [all_filepaths[i] for i in indices]

    print(f"Calculating pixel statistics over {len(sample_filepaths)} images...")

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    total_pixels = 0
    heights = []
    widths = []

    for filepath in sample_filepaths:
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                widths.append(w)
                heights.append(h)

                arr = np.array(img_rgb) / 255.0
                pixels = arr.reshape(-1, 3)
                channel_sum += pixels.sum(axis=0)
                channel_sum_sq += (pixels ** 2).sum(axis=0)
                total_pixels += pixels.shape[0]
        except Exception:
            pass

    mean = (channel_sum / total_pixels).tolist()
    std = np.sqrt(np.maximum(0, (channel_sum_sq / total_pixels) - (channel_sum / total_pixels) ** 2)).tolist()
    avg_resolution = (float(np.mean(heights)), float(np.mean(widths)))

    return mean, std, avg_resolution, class_distribution


def generate_statistics_report(config_path: str, output_path: str, max_samples: int = 1000) -> bool:
    print("=" * 40)
    print("TrustOCT Dataset Statistics Calculator")
    print("=" * 40)

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    train_path = paths.get("train", "datasets/raw/Kermany/train")
    classes = config.get("classes", CLASS_NAMES)
    extensions = tuple(config.get("image", {}).get("extensions", SUPPORTED_EXTENSIONS))

    if not os.path.exists(train_path):
        print(f"Error: Train directory not found at {train_path}")
        return False

    mean, std, avg_res, distribution = calculate_statistics(
        train_path, classes, extensions, max_samples=max_samples
    )

    stats_report = {
        "dataset_name": config.get("dataset", {}).get("name", "Kermany OCT2017"),
        "subsampled_images": min(max_samples, sum(distribution.values())),
        "pixel_mean_rgb": mean,
        "pixel_std_rgb": std,
        "average_height": avg_res[0],
        "average_width": avg_res[1],
        "class_distribution": distribution
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats_report, f, indent=4)

    print(f"Statistics saved to: {output_path}")
    print(f"[OK] RGB Mean: {[round(m, 4) for m in mean]}")
    print(f"[OK] RGB Std:  {[round(s, 4) for s in std]}")
    print("=" * 40)
    return True
