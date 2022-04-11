"""
  Some utility functions useful for MPI parallel programming
"""
from __future__ import print_function

from mpi4py import MPI
import numpy as np
import h5py
import os


# =============================================================================
# I/O Utilities

def pprint(str="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(str + end, end=' ')


def scatter(comm, var):
    # First broadcast the total_size from the master
    total_size = int(comm.bcast(len(var), root=0))

    # Then calculate the counts (size for each worker)
    counts = [total_size // comm.size + 1 if i < total_size % comm.size
              else total_size // comm.size for i in range(comm.size)]

    if comm.rank == 0:
        displs = np.append([0], np.cumsum(counts[:-1]))
        recvbuf = np.empty(counts[comm.rank], dtype=var.dtype.char)
        comm.Scatterv([var, counts, displs, var.dtype.char], recvbuf, root=0)
    else:
        sendbuf = None
        recvbuf = np.empty(counts[comm.rank], dtype=var.dtype.char)
        comm.Scatterv(sendbuf, recvbuf, root=0)

    return recvbuf


def h5_scatter(comm, varname, fname=''):
    # First broadcast the fname from the master
    fname = comm.bcast(fname, root=0)
    assert os.path.exists(fname), f'{fname} does not exist!'

    # Open the file
    with h5py.File(fname, 'r') as f:
        # extract the dataset
        assert varname in f, f'{varname} does not in {fname}!'
        dset = f[varname]
        total_size = len(dset)
        # Then calculate the counts (size for each worker)
        counts = [total_size // comm.size + 1 if i < total_size % comm.size
                  else total_size // comm.size for i in range(comm.size)]
        assert total_size == np.sum(counts), 'Sum of counts and total sum mismatch'
        displs = np.append([0], np.cumsum(counts[:-1]))
        recvbuf = dset[int(displs[comm.rank]):int(displs[comm.rank] + counts[comm.rank])]
    return recvbuf
