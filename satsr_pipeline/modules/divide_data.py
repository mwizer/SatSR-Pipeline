import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split

# ToDo: Typing + docs.


def move_file(src, dst):
    shutil.move(src, dst)


def move_files(file_list, src_dir, dst_dir, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                move_file, os.path.join(src_dir, f), os.path.join(dst_dir, f)
            )
            for f in file_list
        ]
        for future in futures:
            future.result()


def divide_images(
    divide_mode: str,
    download_path: str,
    title: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    workers = min(8, os.cpu_count())
    print(download_path, title)
    img_list = [f for f in os.listdir(download_path) if f.startswith(title)]
    if len(img_list) == 0:
        print("No images to divide")

    elif divide_mode == "train-val":
        train_files, val_files = train_test_split(
            img_list, test_size=(1 - train_ratio), random_state=42
        )
        os.makedirs(os.path.join(download_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(download_path, "val"), exist_ok=True)
        move_files(
            train_files, download_path, os.path.join(download_path, "train"), workers
        )
        move_files(
            val_files, download_path, os.path.join(download_path, "val"), workers
        )

    elif divide_mode == "train-val-test":
        train_files, temp_files = train_test_split(
            img_list, test_size=(1 - train_ratio), random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
        )
        
        os.makedirs(os.path.join(download_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(download_path, "val"), exist_ok=True)
        os.makedirs(os.path.join(download_path, "test"), exist_ok=True)
        move_files(
            train_files, download_path, os.path.join(download_path, "train"), workers
        )
        move_files(
            val_files, download_path, os.path.join(download_path, "val"), workers
        )
        move_files(
            test_files, download_path, os.path.join(download_path, "test"), workers
        )
