import random
import shutil
from pathlib import Path


def copy_random_images(
    src_root: str, dst_dir: str, num_images: int = 10, exts: tuple = (".jpg", ".jpeg", ".png", ".JPEG")
):
    """
    Copy a specified number of random images from a nested source directory to a destination directory.

    The function recursively searches for image files in the source directory (e.g., structured like ImageNet),
    randomly selects a specified number of them, and copies them to the destination directory with renamed files.

    :param src_root: Path to the root directory containing images (possibly organized in subdirectories).
    :param dst_dir: Path to the output directory where the selected images will be copied.
    :param num_images: Number of images to randomly select and copy. Default is 10.
    :param exts: Tuple of allowed image file extensions (case-insensitive). Default is ('.jpg', '.jpeg', '.png').
    """
    src_root = Path(src_root)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(src_root.rglob("*"))
    image_paths = [p for p in all_images if p.suffix.lower() in exts]

    if len(image_paths) < num_images:
        raise ValueError(f"Found only {len(image_paths)} images, but {num_images} requested.")

    selected = random.sample(image_paths, num_images)

    for i, path in enumerate(selected):
        new_name = f"image_{i:02d}{path.suffix.lower()}"
        shutil.copy(path, dst_dir / new_name)
        print(f"Copied: {path} → {dst_dir / new_name}")


if __name__ == "__main__":
    copy_random_images(
        src_root="../../imagenet/ILSVRC/Data/CLS-LOC/train", dst_dir="assets/random_samples100", num_images=100
    )
