from img2dataset import download
import shutil
import sys,os
import time

output_dir = os.path.abspath("bench")

def main():

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    start = time.time()
    print(f'Start it!')
    download(
        processes_count=1,
        thread_count=4,
        url_list="tests/test_files/multifile_test",
        image_size=256,
        output_folder=output_dir,
        output_format="webdataset",
        input_format="parquet-np",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        np_path='tests/test_files/multifile_test',
        # index_path='/export/group/datasets/laion5B/indices/image_PQ128/knn.index000',
        index_path='/export/group/datasets/laion5B/laion2B-en-PQ128-indices/merged-knn.index',
        enable_faiss_memory_mapping=False,
        k=32,
        only_reader=True,
        start_input_file=0,
        max_input_file=1,
        only_ids=False
    )

    print(f'Done! After {(start-time.time()) / 60.} mins.')


# rm -rf bench
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    main()