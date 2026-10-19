import argparse
import io
import os
import random
from pathlib import Path
import zipfile
import requests
from pycocotools.coco import COCO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DEFAULT_DATA_DIR / "annotations"
COCO_UNLABELED_ANN_URL = ("http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip")
ANN_FILE_NAME = "image_info_unlabeled2017.json"


def ensure_annotations(download_dir: Path = DEFAULT_DATA_DIR):
    """Download and extract COCO unlabeled2017 image info if not already present"""
    ann_file = download_dir / "annotations" / ANN_FILE_NAME
    if ann_file.is_file():
        return ann_file

    ann_dir = download_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading unlabeled annotations from {COCO_UNLABELED_ANN_URL}...")
    response = requests.get(COCO_UNLABELED_ANN_URL, stream=True, timeout=60)
    response.raise_for_status()

    #extract annotations directly from the downloaded zip stream
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(download_dir)

    if not ann_file.is_file():
        raise FileNotFoundError(f"Failed to extract {ANN_FILE_NAME} to {ann_file}")

    print(f"Annotations ready at {ann_file}")
    return ann_file


def download_coco_unlabeled(num_images: int = 50,data_dir: Path | str = DEFAULT_DATA_DIR,seed: int | None = None,):
    """
    Download unlabeled images from COCO 2017 into the specified data folder 
    and lists of paths to downloaded images
    """
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    ann_file = ensure_annotations(data_dir)

    print("Loading COCO annotations...")
    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()
    total_available = len(img_ids)
    print(f"Total unlabeled images available in COCO: {total_available}")

    if seed is not None:
        random.seed(seed)

    if 0 < num_images < total_available:
        selected_ids = random.sample(img_ids, num_images)
    else:
        selected_ids = img_ids

    imgs = coco.loadImgs(selected_ids)
    downloaded_paths = []
    total_to_download = len(imgs)

    print(f"Starting download of {total_to_download} images into '{data_dir}'...")

    for i, img_info in enumerate(imgs, start=1):
        file_name = img_info["file_name"]
        coco_url = img_info["coco_url"]
        dest_path = data_dir / file_name

        if dest_path.is_file() and dest_path.stat().st_size > 0:
            print(f"[{i}/{total_to_download}] Skipping existing: {file_name}")
            downloaded_paths.append(dest_path)
            continue

        temp_path = data_dir / f"{file_name}.tmp"
        try:
            with requests.get(coco_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            temp_path.replace(dest_path)
            downloaded_paths.append(dest_path)
            print(f"[{i}/{total_to_download}] Downloaded: {file_name}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"[{i}/{total_to_download}] Error downloading {file_name} from {coco_url}: {e}")

    print(f"\nDone! Successfully prepared {len(downloaded_paths)} images in '{data_dir}'.")
    return downloaded_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download unlabeled images from MS COCO dataset into a data folder."
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=50,
        help="Number of images to download (default: 50). Pass 0 or -1 for all.",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Target directory to store images (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )

    args = parser.parse_args()
    download_coco_unlabeled(
        num_images=args.num_images,
        data_dir=args.data_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

