import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

def move_file(src, dst):
    """
    Move a file from source to destination.

    Parameters
    ----------
    src : str
        The source file path.
    dst : str
        The destination file path.
    """
    shutil.move(src, dst)

def move_files(file_list, src_dir, dst_dir, max_workers):
    """
    Move multiple files from source directory to destination directory using multithreading.

    Parameters
    ----------
    file_list : list
        List of file names to be moved.
    src_dir : str
        The source directory path.
    dst_dir : str
        The destination directory path.
    max_workers : int
        The maximum number of worker threads to use.
    """
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
    """
    Divide images into train, validation, and test sets.

    Parameters
    ----------
    divide_mode : str
        The mode of division. Can be 'train-val' or 'train-val-test'.
    download_path : str
        The path to the directory containing the images.
    title : str
        The title prefix of the images to be divided.
    train_ratio : float, optional
        The ratio of training images. Default is 0.7.
    val_ratio : float, optional
        The ratio of validation images. Default is 0.15.
    test_ratio : float, optional
        The ratio of test images. Default is 0.15.
    """
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