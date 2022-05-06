from img2dataset import download
import shutil
import sys,os


output_dir = os.path.abspath("bench")

def main():

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        processes_count=1,
        thread_count=1,
        url_list="tests/test_files/test_10000_with_index.parquet",
        image_size=256,
        output_folder=output_dir,
        output_format="webdataset",
        input_format="parquet-np",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        np_path='tests/test_files/arr_10000.npy',
        index_path='/export/group/datasets/laion5B/laion5b-index/image.index',
        enable_faiss_memory_mapping=False,
        k=50
    )

# rm -rf bench
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    main()