import faiss
import numpy as np
import time
import os


print('#'*100)
print('Benchmarking search')
print('#'*100)

print(f'Load index 2B')
index = faiss.read_index('/export/group/datasets/laion5B/laion2B-en-PQ128-indices/merged-knn.index')
# index = faiss.read_index('/export/group/datasets/laion5B/laion2B-en-PQ128-indices/knn.index00')
faiss.downcast_index(index.index).make_direct_map()

print('First measuring time to reconstruct single example')
start = time.time()
print(f'Reconstructing single vector took {time.time()-start}')

print('Setting num threads to 128')
faiss.omp_set_num_threads(128)

print('Create test arr')
test_arr = np.random.randn(10000, 768)
test_arr = (test_arr/np.linalg.norm(test_arr,-1,keepdims=True)).astype(np.float32)

def bench(arr, k):
    print('#' * 100)
    print(f'Benchmarking {k} nns for {arr.shape[0]} examples')
    start = time.time()
    # dist, ind, emb = index.search_and_reconstruct(arr,k)
    dist, ind = index.search(arr, k)


    print(f'Search speed {(time.time()-start)/arr.shape[0]} secs/example.')
    print('#' * 100)


for n_ex in [10,100,1000,10000]:

    for k in [20,30,40,50]:

        bench(test_arr[:n_ex],k)


print('Setting num threads to 4')
faiss.omp_set_num_threads(4)

for n_ex in [10,100,1000,10000]:

    for k in [20,30,40,50]:

        bench(test_arr[:n_ex],k)

