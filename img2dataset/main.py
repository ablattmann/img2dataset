"""Img2dataset"""

from typing import List, Optional
import fire
import logging
from .logger import LoggerProcess
from .resizer import Resizer
from .writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from .reader import Reader, PreExReader
from .downloader import Downloader, DummyDownloader
from .distributor import multiprocessing_distributor, pyspark_distributor
import fsspec
import sys
import signal
import os

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)


def download2(url_list: str, # this is the path to the meta data
    image_size: int = 256,
    output_folder: str = "images",
    processes_count: int = 1,
    resize_mode: str = "border",
    resize_only_if_bigger: bool = False,
    upscale_interpolation: str = "lanczos",
    downscale_interpolation: str = "area",
    encode_quality: int = 95,
    skip_reencode: bool = False,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    thread_count: int = 256,
    number_sample_per_shard: int = 10000,
    extract_exif: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    enable_wandb: bool = False,
    wandb_project: str = "img2dataset",
    oom_shard_count: int = 5,
    compute_md5: bool = True,
    distributor: str = "multiprocessing",
    subjob_size: int = 1000,
    retries: int = 0,
    disable_all_reencoding: bool = False,
    np_path = None,
    index_path=None,
    enable_faiss_memory_mapping=False,
    k=20,
    start_input_file=0,
    max_input_file=None,
    benchmark=False,
    only_ids=False,):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())


    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    # logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, config_parameters)

    tmp_path = output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)

    # if not only_reader:
    #     print('Setting up signal handler to clean up tmp_dir')
    #     def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
    #         try:
    #             fs.rm(tmp_dir, recursive=True)
    #         except Exception as _:  # pylint: disable=broad-except
    #             pass
    #         logger_process.terminate()
    #         sys.exit(0)
    #
    #     signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    start_shard_id = 0

    # if not fs.exists(output_path):
    #     fs.mkdir(output_path)
    #     start_shard_id = 0
    # else:
    #     existing_top_level_files = [x for x in fs.glob(output_path + "/*") if x != tmp_dir and "stats" not in x]
    #     if len(existing_top_level_files) == 0:
    #         start_shard_id = 0
    #     else:
    #         start_shard_id = (
    #                 max([int(x.split("/")[-1].split(".")[0]) for x in existing_top_level_files if x != tmp_dir]) + 1
    #         )

    # logger_process.start_shard_id = start_shard_id
    # logger_process.start()

    reader = Reader(
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        start_shard_id,
        tmp_path,
        np_path=np_path,
        index=index_path,
        enable_faiss_memory_mapping=enable_faiss_memory_mapping,
        k=k,
        start_input_file=start_input_file,
        max_input_file=max_input_file,
        benchmark=benchmark,
        only_ids=only_ids,
        is_dummy=False
    )




    for r in reader:
        shard_id, fname = r
        print(f'Wrote tmp files for shard #{shard_id} to {fname}')
    # if not only_reader:
    #     fs.rm(tmp_dir, recursive=True)

def download(
    url_list: str, # this is the path to the meta data
    image_size: int = 256,
    output_folder: str = "images",
    processes_count: int = 1,
    resize_mode: str = "border",
    resize_only_if_bigger: bool = False,
    upscale_interpolation: str = "lanczos",
    downscale_interpolation: str = "area",
    encode_quality: int = 95,
    skip_reencode: bool = False,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    thread_count: int = 256,
    number_sample_per_shard: int = 10000,
    extract_exif: bool = True,
    save_additional_columns: Optional[List[str]] = None,
    timeout: int = 10,
    enable_wandb: bool = False,
    wandb_project: str = "img2dataset",
    oom_shard_count: int = 5,
    compute_md5: bool = True,
    distributor: str = "multiprocessing",
    subjob_size: int = 1000,
    retries: int = 0,
    disable_all_reencoding: bool = False,
    np_path = None,
    index_path=None,
    enable_faiss_memory_mapping=False,
    k=20,
    only_reader=False,
    start_input_file=0,
    max_input_file=None,
    benchmark=False,
    only_ids=False,
    only_downloader=False
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = dict(locals())

    if only_downloader:
        assert not only_reader

    if only_reader:
        assert not only_downloader

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, config_parameters)

    tmp_path = output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)

    # if not only_reader:
    #     print('Setting up signal handler to clean up tmp_dir')
    #     def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
    #         try:
    #             fs.rm(tmp_dir, recursive=True)
    #         except Exception as _:  # pylint: disable=broad-except
    #             pass
    #         logger_process.terminate()
    #         sys.exit(0)
    #
    #     signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    start_shard_id = 0

    # if not fs.exists(output_path):
    #     fs.mkdir(output_path)
    #     start_shard_id = 0
    # else:
    #     existing_top_level_files = [x for x in fs.glob(output_path + "/*") if x != tmp_dir and "stats" not in x]
    #     if len(existing_top_level_files) == 0:
    #         start_shard_id = 0
    #     else:
    #         start_shard_id = (
    #                 max([int(x.split("/")[-1].split(".")[0]) for x in existing_top_level_files if x != tmp_dir]) + 1
    #         )

    logger_process.start_shard_id = start_shard_id
    logger_process.start()

    reader = Reader(
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        start_shard_id,
        tmp_path,
        np_path=np_path,
        index=index_path,
        enable_faiss_memory_mapping=enable_faiss_memory_mapping,
        k=k,
        start_input_file=start_input_file,
        max_input_file=max_input_file,
        benchmark=benchmark,
        only_ids=only_ids,
        is_dummy=only_downloader
    )

    if only_downloader:
        print('Overwrite reader')
        reader = PreExReader(output_folder,
                             column_list=reader.column_list,
                             only_ids=reader.only_ids)

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    elif output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore
    elif output_format == "tfrecord":
        sample_writer_class = TFRecordSampleWriter  # type: ignore
    elif output_format == "dummy":
        sample_writer_class = DummySampleWriter  # type: ignore

    resizer = Resizer(
        image_size=image_size,
        resize_mode=resize_mode,
        resize_only_if_bigger=resize_only_if_bigger,
        upscale_interpolation=upscale_interpolation,
        downscale_interpolation=downscale_interpolation,
        encode_quality=encode_quality,
        skip_reencode=skip_reencode,
        disable_all_reencoding=disable_all_reencoding,
    )

    if only_reader:
        downloader = DummyDownloader()
    else:
        downloader = Downloader(
            sample_writer_class=sample_writer_class,
            resizer=resizer,
            thread_count=thread_count,
            save_caption=save_caption,
            extract_exif=extract_exif,
            output_folder=output_folder,
            column_list=reader.column_list,
            timeout=timeout,
            number_sample_per_shard=number_sample_per_shard,
            oom_shard_count=oom_shard_count,
            compute_md5=compute_md5,
            retries=retries,
            load_np=np_path is not None,
            only_ids=reader.only_ids
        )


    print("Starting the downloading of this file")
    if distributor == "multiprocessing":
        distributor_fn = multiprocessing_distributor
    elif distributor == "pyspark":
        distributor_fn = pyspark_distributor
    else:
        raise ValueError(f"Distributor {distributor} not supported")

    distributor_fn(
        processes_count,
        downloader,
        reader,
        subjob_size,
    )
    logger_process.join()
    # if not only_reader:
    #     fs.rm(tmp_dir, recursive=True)


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
