# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import sys

import numpy as np
# import h5py
import psutil
import argparse
from mpi4py import MPI
from parutils import pprint, scatter
from memory_profiler import memory_usage
# from memory_profiler import profile

# max memory to use, in bytes
MEMORY_LIM = 200
COMPRESSION = int(0)
CHUNK_SIZE = int(0)

parser = argparse.ArgumentParser(
    description='Script to evaluate memory footprint of h5py particle distribution generation')

parser.add_argument('-m', '--memory', type=float, default=MEMORY_LIM,
                    help='Maximum memory to allocate for the particles, in MB. Default: 200')

parser.add_argument('-gzip', '--gzip', type=int, default=COMPRESSION,
                    help='Compression level (0-9). Default: 0')

parser.add_argument('-c', '--chunk', type=int, default=CHUNK_SIZE,
                    help='H5PY chunk size. Default: 1e6')

parser.add_argument('-o', '--outdir', type=str, default='./',
                    help='Directory to write output.')

# @profile
def use_numpy(num_elems):
    dt = np.arange(np.int64(num_elems)).astype(dtype=np.float64)
    dE = np.arange(np.int64(num_elems)).astype(dtype=np.float64)
    idx = np.arange(np.int64(num_elems), dtype=np.int64)
    # print(f'dt original avg: {np.mean(dt[:])}')
    # print(f'dE original avg: {np.mean(dE[:])}')
    # print(f'id original avg: {np.mean(idx[:])}')
    return dt, dE, idx


# @profile(stream=open(f'memory-profile-{comm.rank}.txt','w'))
def generate_scatter_data(comm, total_elems, args):
    if comm.rank == 0:
        dt, dE, idx = use_numpy(total_elems)
        dt = scatter(comm, dt)
        dE = scatter(comm, dE)
        idx = scatter(comm, idx)
    else:
        dt = np.array([], dtype=np.float64)
        dE = np.array([], dtype=np.float64)
        idx = np.array([], dtype=np.int64)
        dt = scatter(comm, dt)
        dE = scatter(comm, dE)
        idx = scatter(comm, idx)

    print(f'[{comm.rank}] dt Len: {len(dt)}, Avg: {np.mean(dt)}')
    print(f'[{comm.rank}] dE Len: {len(dE)}, Avg: {np.mean(dE)}')
    print(f'[{comm.rank}] id Len: {len(idx)}, Avg: {np.mean(idx)}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pprint('----- Testing NUMPY ------')
    args = parser.parse_args()
    args.memory *= 1e6

    memory = psutil.virtual_memory()
    pprint('Virtual memory ' + str(memory))
    swap = psutil.swap_memory()
    pprint('Swap memory ' + str(swap))
    # basically we need dt, de and id
    # assuming float64 and int64
    elem_size = (2 * 8 + 8)
    pprint(f'Max points that can be generated (assuming 0 OS memory): {memory.total // elem_size / 1e6} M')
    pprint(f'Max points that cab be generated (with current utilization): {memory.available // elem_size / 1e6} M')
    num_elems = args.memory // elem_size
    pprint(f'Points per worker according to mem limitation ({args.memory // 1e6}MB): {num_elems / 1e6} M')

    comm = MPI.COMM_WORLD
    total_elems = comm.size * num_elems

    pprint(f'Total number of points that will be generated ({total_elems * elem_size / 1e6}MB) {total_elems/1e6} M')

    pprint("-" * 78)
    pprint(" Running on %d cores" % comm.size)
    pprint("-" * 78)
    print('[{}]@{}: Hello World!'.format(comm.rank, MPI.Get_processor_name()))

    # Everything goes in a function, for easier reporting
    mem_usage = memory_usage((generate_scatter_data, (comm, total_elems, args), {}))
    # generate_scatter_data(comm, total_elems, args)

    # print(f'[{comm.rank}] Memory footprint: {np.max(mem_usage) - np.min(mem_usage)} MB')
    print(f'[{comm.rank}] Memory footprint: {np.max(mem_usage)} MB')

