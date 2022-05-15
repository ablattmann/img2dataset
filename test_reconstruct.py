import faiss
from faiss.contrib.ondisk import merge_ondisk
import numpy as np
import time
import os


print('#'*100)
print('Benchmarking search')
print('#'*100)

# print(f'Load index 2B')
# # index = faiss.read_index('/export/group/datasets/laion5B/laion2B-en-PQ128-indices/merged-knn.index')
# faiss.downcast_index(index.index).make_direct_map()
#
# print('Setting num threads to 128')
# faiss.omp_set_num_threads(128)
#
# print('Create test arr')
test_arr = np.random.randn(10, 768)
test_arr = (test_arr/np.linalg.norm(test_arr,-1,keepdims=True)).astype(np.float32)
#
# _, _ , embs = index.search_and_reconstruct(test_arr,k=40)
#
# _, ids = index.search(test_arr,k=40)
# print(ids)
# start = time.time()
# rec_emb = index.reconstruct(int(ids[0,0]))
# print(f'Reconstruction took {(time.time()-start)} secs')
#
# print('Via search, then reconstruct')
# print(rec_emb.shape,np.amin(rec_emb),np.amax(rec_emb))
#
# print('Via search_and_reconstruct')
# test_emb = embs[0,0]
# print(test_emb.shape,np.amin(test_emb),np.amax(test_emb))
#
# print('Hops ins Wasser!')
# assert np.equal(rec_emb,test_emb).all(), 'Not right'

print('#'*100)
print('IVF index')
block_name = ['/export/group/datasets/laion5B/laion2B-en-PQ128-indices/knn.index00']
empty = faiss.read_index(block_name[0],faiss.IO_FLAG_MMAP)
empty.ntotal=0

dir = '/export/group/datasets/laion5B/from_m_test'
os.makedirs(dir,exist_ok=True)
merge_ondisk(empty,block_name,dir+'/merged_index.ivfdata')
faiss.write_index(empty,dir+'/populated.index')
index = faiss.read_index(dir + '/populated.index',faiss.IO_FLAG_ONDISK_SAME_DIR)
faiss.downcast_index(index.index).make_direct_map()
print('Search')
_, ids = index.search(test_arr,k=40)
print(ids)
start = time.time()
rec_emb = index.reconstruct(int(ids[0,0]))
print(f'Reconstruction took {(time.time()-start)} secs')