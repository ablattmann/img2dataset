"""Reader is module to read the url list and return shards"""
import os
from multiprocessing.pool import ThreadPool
import math
import fsspec
import time
import pyarrow.parquet as pq
import pyarrow.csv as csv_pq
import pyarrow as pa
import pandas as pd
import numpy as np
import faiss
from collections import defaultdict
from glob import glob


def connected_components(neighbors):
    """find connected components in the graph"""
    seen = set()

    def component(node):
        r = []
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= set(neighbors[node]) - seen
            r.append(node)
        return r

    u = []
    for node in neighbors:
        if node not in seen:
            u.append(component(node))
    return u

def get_non_uniques(embeddings, threshold=0.94):
    """find non-unique embeddings"""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # pylint: disable=no-value-for-parameter
    l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

    same_mapping = defaultdict(list)

    # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
    for i in range(embeddings.shape[0]):
        for j in I[l[i]: l[i + 1]]:
            same_mapping[int(i)].append(int(j))

    groups = connected_components(same_mapping)
    non_uniques = set()
    for g in groups:
        for e in g[1:]:
            non_uniques.add(e)

    return list(non_uniques)



def load_index(path, enable_faiss_memory_mapping):
    if enable_faiss_memory_mapping:
        if os.path.isdir(path):
            return faiss.read_index(path + "/populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    else:
        assert os.path.isdir(path)
        index_files = glob(path.rstrip('/') + '/*.index*')
        print('Read index')
        return faiss.read_index(index_files)

class Reader:
    """
    The reader class reads an url list and returns shards
    It provides an iter method
    It provides attributes:
    - column_list: the list of columns to read
    - input_format: the format of the input file
    - url_col: the column name of the url
    - caption_col: the column name of the caption
    - save_additional_columns: the list of additional columns to save
    - number_sample_per_shard: the number of samples per shard
    - start_shard_id: the id of the first shard
    """

    def __init__(
        self,
        url_list,
        input_format,
        url_col, # should be URL
        caption_col, # should be TEXT
        save_additional_columns,
        number_sample_per_shard,
        start_shard_id,
        tmp_path,
        np_path = None,
        index=None,
        enable_faiss_memory_mapping=False,
        k=None
    ) -> None:
        self.input_format = input_format
        if self.input_format == 'parquet-np':
            assert np_path is not None
            assert index is not None
            assert k is not None
        self.k = k+1 if k else k
        self.enable_faiss_mmap = enable_faiss_memory_mapping
        self.index = index
        self.np_path = np_path
        self.url_col = url_col
        self.caption_col = caption_col
        self.save_additional_columns = save_additional_columns
        self.number_sample_per_shard = number_sample_per_shard
        self.start_shard_id = start_shard_id

        fs, url_path = fsspec.core.url_to_fs(url_list)
        self.fs = fs
        self.tmp_path = tmp_path

        if fs.isdir(url_path):
            self.input_files = sorted(fs.glob(url_path + "/*." + input_format if input_format != 'parquet-np' else 'parquet'))
        else:
            self.input_files = [url_path]

        if self.np_path is not None:
            np_fs, npp = fsspec.core.url_to_fs(self.np_path)
            self.np_fs = np_fs
            if np_fs.isdir(self.np_path):
                self.np_files = sorted(np_fs.glob(npp + '/*.npy'))
            else:
                self.np_files = [self.np_path]
        else:
            self.np_files = [None] * len(self.input_files)

        if self.input_format == "txt":
            self.column_list = ["url"]
        elif self.input_format in ["json", "csv", "tsv", "tsv.gz", "parquet", 'parquet-np']:
            self.column_list = self.save_additional_columns if self.save_additional_columns is not None else []
            if self.caption_col is not None:
                self.column_list = self.column_list + ["caption", "url"]
            else:
                self.column_list = self.column_list + ["url"]


        tes = 1

    def _save_to_arrow(self, input_file, np_file=None):
        """Read the input file and save to arrow files in a temporary directory"""
        embeddings = None
        if self.input_format in ["txt", "json", "csv", "tsv"]:
            with self.fs.open(input_file, mode="rb") as file:
                if self.input_format == "txt":
                    df = csv_pq.read_csv(file, read_options=csv_pq.ReadOptions(column_names=["url"]))
                elif self.input_format == "json":
                    df = pa.Table.from_pandas(pd.read_json(file))
                elif self.input_format == "csv":
                    df = csv_pq.read_csv(file)
                elif self.input_format == "tsv":
                    df = csv_pq.read_csv(file, parse_options=csv_pq.ParseOptions(delimiter="\t"))
                else:
                    raise ValueError(f"Unknown input format {self.input_format}")
        elif self.input_format == "tsv.gz":
            with self.fs.open(input_file, encoding="utf-8", mode="rb", compression="gzip") as file:
                df = csv_pq.read_csv(file, parse_options=csv_pq.ParseOptions(delimiter="\t"))
        elif self.input_format == "parquet":
            with self.fs.open(input_file, mode="rb") as file:
                columns_to_read = [self.url_col]
                if self.caption_col is not None:
                    columns_to_read += [self.caption_col]
                if self.save_additional_columns is not None:
                    columns_to_read += self.save_additional_columns
                df = pq.read_table(file, columns=columns_to_read)
        elif self.input_format == 'parquet-np':
            with self.fs.open(input_file, mode="rb") as file:
                columns_to_read = [self.url_col]
                if self.caption_col is not None:
                    columns_to_read += [self.caption_col]
                if self.save_additional_columns is not None:
                    columns_to_read += self.save_additional_columns
                df = pq.read_table(file, columns=columns_to_read)

            # embeddings are assumed to be aligned
            embeddings = np.load(np_file,mmap_mode='r')

            print(f'Loading index from {self.index}')
            nn_index = load_index(self.index,enable_faiss_memory_mapping=self.enable_faiss_mmap)
            print('Done loading index')

        else:
            raise ValueError(f"Unknown input format {self.input_format}")

        column_names = df.column_names
        if self.caption_col is not None:
            column_names = [c if c != self.caption_col else "caption" for c in column_names]
        column_names = [c if c != self.url_col else "url" for c in column_names]

        df = df.rename_columns(column_names)

        number_samples = df.num_rows

        number_shards = math.ceil(df.num_rows / self.number_sample_per_shard)

        def write_shard(shard_id):
            begin_shard = shard_id * self.number_sample_per_shard
            end_shard = min(number_samples, (1 + shard_id) * self.number_sample_per_shard)
            df_shard = df.slice(begin_shard, end_shard - begin_shard).select([cll for cll in self.column_list if cll not in ['nn_indices','nn_embeddings']])

            indices = None
            if embeddings is not None:
                emb_shard = embeddings[begin_shard:end_shard]
                emb_shard_normalize = emb_shard / np.linalg.norm(emb_shard,axis=-1,keepdims=True)
                print('Start search')
                start = time.time()
                distances, indices, nn_embeddings = nn_index.search_and_reconstruct(x=emb_shard_normalize.astype(np.float32),
                                                                                  k=self.k)
                print(f'End search, took {time.time() - start} s')
                # remove first entry as this is likely to be the image embeddings itself
                nn_embeddings = nn_embeddings[:,1:]
                indices = indices[:,1:]
                # indices = indices
                # nn_embeddings = nn_embeddings
                nb_results = np.nonzero(indices != -1)
                max_row = np.amax(nb_results[0])

                result_nn_embeddings = []
                result_ids = []
                for row in range(max_row+1):
                    r_elements = nb_results[1][nb_results[0]==row]
                    if r_elements.size >0:
                        filtered = nn_embeddings[row,r_elements]
                        ind_filtered = indices[row,r_elements]
                        to_remove = set(get_non_uniques(filtered))
                        to_keep = np.logical_not(np.isin(np.arange(filtered.shape[0]),list(to_remove)))
                        filtered = filtered[to_keep]
                        ind_filtered = filtered[to_keep]
                        # pad with zeros
                        if filtered.shape[0] < nn_embeddings.shape[1]:
                            filtered = np.concatenate([filtered]+[np.full((1,filtered.shape[-1]),
                                                                          -1.)]*(nn_embeddings.shape[1]-filtered.shape[0]))
                            ind_filtered = np.concatenate([ind_filtered,np.full((indices.shape[-1]-ind_filtered,))])
                    else:
                        filtered = np.full_like(nn_embeddings,-1.)
                        ind_filtered = np.full_like(indices, -1, dtype=np.int32)

                    result_nn_embeddings.append(filtered)
                    result_ids.append(ind_filtered)

                nn_embeddings = np.stack(result_nn_embeddings,axis=0)
                indices = np.stack(result_ids,axis=0)
                print(nn_embeddings.shape, indices.shape)

                # for dim_id, batched_component in enumerate(emb_shard.transpose(1,0)):
                #     df_shard = df_shard.append_column(str(dim_id),pa.array(batched_component))

            tmp_file = self.tmp_path + f"/{shard_id + self.start_shard_id}.feather"
            np_file = self.tmp_path + f'/{shard_id + self.start_shard_id}.npz'
            # emb_file = self.tmp_path      + f'/{shard_id + self.start_shard_id}.npy'
            # ind_file = self.tmp_path + f'/{shard_id + self.start_shard_id}_indices.npy'


            for i in range(10):
                try:

                    # potentially alternative way to save np data
                    # if nn_embeddings is not None:
                    #     fs_np, tmp_np_path = fsspec.core.url_to_fs(emb_file)
                    #     with fs_np.open(tmp_np_path, 'wb') as np_f:
                    #         npb = BytesIO()
                    #         np.save(npb, nn_embeddings)
                    #         np_f.write(npb.getbuffer())
                    #
                    # if indices is not None:
                    #     fs_np, tmp_np_path = fsspec.core.url_to_fs(ind_file)
                    #     with fs_np.open(tmp_np_path, 'wb') as np_f:
                    #         npb = BytesIO()
                    #         np.save(npb, indices)
                    #         np_f.write(npb.getbuffer())


                    fs, tmp_path = fsspec.core.url_to_fs(tmp_file)
                    if indices is not None:
                        np.savez(np_file,nn_indices=indices, nn_embeddings=nn_embeddings)
                    with fs.open(tmp_path, "wb") as file:
                        with pa.ipc.new_file(file, df_shard.schema) as writer:
                            writer.write_table(df_shard)
                    return (shard_id, tmp_file)
                except Exception as e:  # pylint: disable=broad-except
                    if i != 9:
                        print("retrying to write to file due to error:", e)
                        time.sleep(1)
                    else:
                        raise e
            # can't reach here
            raise Exception("Failed to write to file.")

        for i in range(10):
            shards = []
            # thread pool to make it faster to write files to low latency file systems (ie s3, hdfs)
            try:
                with ThreadPool(32) as thread_pool:
                    for shard in thread_pool.imap_unordered(write_shard, range(number_shards)):
                        shards.append(shard)
                break
            except Exception as e:  # pylint: disable=broad-except
                raise e
                if i != 9:
                    print("retrying whole sharding to write to files due to error:", e)
                    time.sleep(2 * i)
                else:
                    raise e

        shards.sort(key=lambda k: k[0])

        del df

        return shards

    def __iter__(self):
        """
        Iterate over shards, yield shards of size number_sample_per_shard or less for the last one
        Each shard is a tuple (shard_id, shard)
        shard is a tuple (sample id, sample)
        sample is a tuple of the columns
        """
        for i, (input_file, np_file) in enumerate(zip(self.input_files,self.np_files)):
            info = "Sharding file number " + str(i + 1) + " of " + str(len(self.input_files)) + " called " + input_file

            if np_file is not None:
                info += f' and aligned embeddings at {np_file}'
            print(info)
            shards = self._save_to_arrow(input_file, np_file)
            print("File sharded in " + str(len(shards)) + " shards")
            print(
                "Downloading starting now, check your bandwidth speed (with bwm-ng)"
                "your cpu (with htop), and your disk usage (with iotop)!"
            )

            num_shard = 0
            for num_shard, arrow_file in shards:
                yield (
                    num_shard + self.start_shard_id,
                    arrow_file,
                )

                num_shard += 1
            self.start_shard_id += num_shard
