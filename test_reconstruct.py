import faiss
import numpy as np
import time
import os

print('#'*100)
print('Benchmarking search')
print('#'*100)

print(f'Load index 2B')
# index = faiss.read_index('/export/group/datasets/laion5B/laion2B-en-PQ128-indices/merged-knn.index')
index = faiss.read_index('/export/group/datasets/laion5B/laion2B-en-PQ128-indices/knn.index00')
faiss.downcast_index(index.index).make_direct_map()

print('Setting num threads to 128')
faiss.omp_set_num_threads(128)

print('Create test arr')
test_arr = np.random.randn(10, 768)
test_arr = (test_arr/np.linalg.norm(test_arr,-1,keepdims=True)).astype(np.float32)

_, _ , embs = index.search_and_reconstruct(test_arr,k=40)

_, ids = index.search(test_arr,k=40)
print(ids)
rec_emb = index.reconstruct(int(ids[0,0]))

print('Via search, then reconstruct')
print(rec_emb.shape,np.amin(rec_emb),np.amax(rec_emb))

print('Via search_and_reconstruct')
test_emb = embs[0,0]
print(test_emb.shape,np.amin(test_emb),np.amax(test_emb))

print('Hops ins Wasser!')
assert np.equal(rec_emb,test_emb).all(), 'Not right'