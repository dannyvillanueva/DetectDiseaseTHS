import multiprocessing
import time

"""
def worker(method, val1, val2, send_end):
    result = val1+val2
    send_end.send(result)

if __name__ == '__main__':
    jobs = []
    pipe_list = []
    for i in range(3):
        recv, send = multiprocessing.Pipe(False)
        if i == 0:
            p = multiprocessing.Process(target=worker, args=(0, 1, 1, send))
        if i == 1:
            p = multiprocessing.Process(target=worker, args=(1, 10, 10, send))
        if i == 2:
            p = multiprocessing.Process(target=worker, args=(2, 5, 5, send))
        jobs.append(p)
        pipe_list.append(recv)
        p.start()
    result_list = [x.recv() for x in pipe_list]
    print(result_list)
"""

import numpy as np
from ths.utils.similarity import Frobenius_Distance, TriUL_sim
from itertools import product

def distance(a, b):
    a = np.array(a)
    b = np.array(b)
    f = Frobenius_Distance(a, b)
    t = TriUL_sim(a, b)
    print("Distance: ",round(f,3), round(t,3))
    return [a, b]


if __name__ == '__main__':
    p = multiprocessing.Pool(processes = 4)
    #start = time.time()
    data = [[[1,1], [2,2]], [[3,3], [4,4]], [[3,3],[1,1]], [[3,3], [7,7]]]
    a = product(data, [[1, 1]])
    #results = p.starmap(distance, product(data, data[0]))
    #p.close()
    #p.join()
    #print("Complete")
    #end = time.time()
    #print('total time (s)= ' + str(end-start))
