import tqdm
import time
import random
import numpy as np


n = 1000000

# 2.1s
start = time.time()
rand = random.random
for i in tqdm.tqdm(range(n)):
    for j in range(4):
        # random.randint(0, 3705)
        int(3706 * rand())
end = time.time()
print(end - start)

# 6s
# start = time.time()
# rand = np.random.randint
# for i in tqdm.tqdm(range(n)):
#     for j in range(4):
#         # np.random.randint(3706)
#         random.randint(0, 3705)
# end = time.time()
# print(end - start)

# 30s
# start = time.time()
# rand = np.random.randint
# for i in tqdm.tqdm(range(n)):
#     for j in range(4):
#         np.random.randint(3706)
#         #rand(3706)
# end = time.time()
# print(end - start)

# 30s
# start = time.time()
# rand = np.random.randint
# for i in tqdm.tqdm(range(n)):
#     for j in range(4):
#         # np.random.randint(3706)
#         rand(3706)
# end = time.time()
# print(end - start)


