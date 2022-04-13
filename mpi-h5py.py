# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import sys

import numpy as np
import h5py
import psutil
import argparse
from mpi4py import MPI
from parutils import pprint, h5_scatter
from memory_profiler import memory_usage
import os
# from memory_profiler import profile

# max memory to use, in bytes
MEMORY_LIM = 200
COMPRESSION = int(0)
CHUNK_SIZE = int(1e6)

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
def use_h5py(num_elems: int, chunk_size: int = 1000000, compression: int = 4, outdir: str = './'):
    fname = f'{outdir}/test-elems{total_elems}-chunk{chunk_size}-compression{compression}.hdf5'
    with h5py.File(fname, 'w') as file:
        dt_dset = file.create_dataset("dt", num_elems, dtype=np.float64, chunks=chunk_size, compression='gzip',
                                      compression_opts=compression)
        dE_dset = file.create_dataset("dE", num_elems, dtype=np.float64, chunks=chunk_size, compression='gzip',
                                      compression_opts=compression)
        id_dset = file.create_dataset("id", num_elems, dtype=np.int32, chunks=chunk_size, compression='gzip',
                                      compression_opts=compression)

        idx = 0
        while idx < num_elems:
            write_elems = min(chunk_size, num_elems - idx)
            dt_dset[idx:idx + write_elems] = np.arange(idx, idx + write_elems).astype(dtype=np.float64)
            dE_dset[idx:idx + write_elems] = np.arange(idx, idx + write_elems).astype(dtype=np.float64)
            id_dset[idx:idx + write_elems] = np.arange(idx, idx + write_elems).astype(dtype=np.int32)
            idx += chunk_size
        # print(f'dt original avg: {np.mean(dt_dset[:])}')
        # print(f'dE original avg: {np.mean(dE_dset[:])}')
        # print(f'id original avg: {np.mean(id_dset[:])}')
    return fname

# @profile
def generate_scatter_data(comm, total_elems, args):
    if comm.rank == 0:
        fname = use_h5py(total_elems, args.chunk, args.gzip, args.outdir)
        dt = h5_scatter(comm, 'dt', fname=fname)
        dE = h5_scatter(comm, 'dE', fname=fname)
        idx = h5_scatter(comm, 'id', fname=fname)
        # f.close()
    else:
        dt = h5_scatter(comm, 'dt')
        dE = h5_scatter(comm, 'dE')
        idx = h5_scatter(comm, 'id')

    print(f'[{comm.rank}] Len: {len(dt)}, Avg: {np.mean(dt)}')
    print(f'[{comm.rank}] Len: {len(dE)}, Avg: {np.mean(dE)}')
    print(f'[{comm.rank}] Len: {len(idx)}, Avg: {np.mean(idx)}')
    # Delete fname
    if comm.rank == 0:
        if os.path.isfile(fname):
            os.remove(fname)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pprint('----- Testing H5PY ------')
    args = parser.parse_args()
    args.memory *= 1e6

    memory = psutil.virtual_memory()
    pprint('Virtual memory ' + str(memory))
    swap = psutil.swap_memory()
    pprint('Swap memory ' + str(swap))
    # basically we need dt, de and id
    # assuming float64 and int32
    elem_size = (2 * 8 + 4)
    pprint(f'Max points that can be generated (assuming 0 OS memory): {memory.total // elem_size / 1e6} M')
    pprint(f'Max points that cab be generated (with current utilization): {memory.available // elem_size / 1e6} M')
    num_elems = int(args.memory // elem_size)
    pprint(f'Points per worker according to mem limitation ({args.memory // 1e6}MB): {num_elems / 1e6} M')

    comm = MPI.COMM_WORLD
    total_elems = comm.size * num_elems

    pprint(f'Total number of points that will be generated ({total_elems * elem_size / 1e6}MB) {total_elems/1e6} M')

    pprint("-" * 78)
    pprint(" Running on %d cores" % comm.size)
    pprint("-" * 78)
    print('[{}]@{}: Hello World!'.format(comm.rank, MPI.Get_processor_name()))

    # Everything goes in a function, for easier reporting
    mem_usage = memory_usage((generate_scatter_data, (comm, total_elems, args), {}), interval=0.1)
    # generate_scatter_data(comm, total_elems, args)

    print(f'[{comm.rank}] Memory footprint: {np.max(mem_usage) - np.min(mem_usage)} MB')



