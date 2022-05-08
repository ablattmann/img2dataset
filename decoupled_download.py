from img2dataset import download
import shutil
import sys,os
import time
# import faiss
import fire

# output_dir = os.path.abspath("bench")
#


def main(output_dir, num_threads, num_proc,  n_sample_per_shard=10000,
         only_ids=False,start_file=0,max_file=None,
         index_path='/export/group/datasets/laion5B/laion2B-en-PQ128-indices/merged-knn.index', faiss_mmap=False,
         url_list="/export/group/datasets/laion5B/aligned_sorted_filtered_metadata_train",
         np_path='/export/group/datasets/laion5B/train_embeddings_s512_filtered'):

    start = time.time()
    print(f'Start it!')
    download(
        processes_count=num_proc,
        thread_count=num_threads,
        url_list=url_list,
        image_size=512,
        output_folder=output_dir,
        output_format="webdataset",
        input_format="parquet-np",
        resize_mode='keep_ratio',
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=n_sample_per_shard,
        distributor="multiprocessing",
        np_path=np_path,
        # index_path='/export/group/datasets/laion5B/indices/image_PQ128/knn.index000',
        index_path=index_path   ,
        enable_faiss_memory_mapping=faiss_mmap,
        k=40,
        only_reader=False,
        start_input_file=start_file,
        max_input_file=max_file,
        only_ids=only_ids,
        only_downloader=True,
    )

    print(f'Done! After {(start-time.time()) / 60.} mins.')


# rm -rf bench
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    fire.Fire(main)