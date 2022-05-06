from img2dataset import download
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

from pyspark import SparkConf, SparkContext


def create_spark_session():
    # this must be a path that is available on all worker nodes
    pex_file = "/export/group/datasets/laion5B/img2dataset/img2dataset.pex"

    os.environ['PYSPARK_PYTHON'] = pex_file
    spark = (
        SparkSession.builder
            .config("spark.submit.deployMode", "client")
            # .config("spark.files", pex_file) \ # you may choose to uncomment this option if you want spark to automatically download the pex file, but it may be slow
            .config("spark.executorEnv.PEX_ROOT", "./.pex")
            # .config("spark.executor.cores", "2")
            # .config("spark.cores.max", "200") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
            .config("spark.driver.port", "5678")
            .config("spark.driver.blockManager.port", "6678")
            .config("spark.driver.host", "master_node")
            .config("spark.driver.bindAddress", "master_node")
            .config("spark.executor.memory", "16GB")  # make sure to increase this if you're using more cores per executor
            .config("spark.executor.memoryOverhead", "8GB")
            .config("spark.task.maxFailures", "100")
            .master("spark://master_node:7077")  # this should point to your master node, if using the tunnelling version, keep this to localhost
            .appName("spark-stats")
            .getOrCreate()
    )
    return spark


output_dir = "/bench"

spark = create_spark_session()

url_list = "some_file.parquet"

download(
    processes_count=1,
    thread_count=32,
    retries=0,
    url_list="tests/test_files/test_10000_with_index.parquet",
    image_size=256,
    output_folder=output_dir,
    output_format="webdataset",
    input_format="parquet-np",
    url_col="URL",
    caption_col="TEXT",
    enable_wandb=False,
    number_sample_per_shard=100,
    distributor="pyspark",
    save_additional_columns=["NSFW", "similarity", "LICENSE"],
    np_path='tests/test_files/arr_10000.npy',
    index_path='/export/group/datasets/laion5B/laion5b-index/image.index',
    enable_faiss_memory_mapping=True,
    k=50
)