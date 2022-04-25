import os
import csv
# import sys
# import fnmatch
import numpy as np
# import subprocess
import argparse
import glob
import re


parser = argparse.ArgumentParser(description='Generate a csv report from the input raw data.',
                                 usage='python extract.py -i [indir]')

# parser.add_argument('-o', '--outfile', type=str, default='stdout',
#                     help='The file to save the report.'
#                     ' Default: (indir)-report.csv')

parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory containing the collected data.')


if __name__ == '__main__':
    args = parser.parse_args()
    indirs = glob.glob(args.indir)
    regexp1 = re.compile(r'\[(\d+)\].*:(.*)MB')
    header = ['exe', 'mem', 'gzip', 'chunk', 'N', 'wpn', 'mpi', 'ftprint0', 'ftprintn_std']
    for indir in indirs:
        print(f'Working on {indir} directory')
        print('\n-------- Generating reports -------\n')
        dic = {}
        for dirs, subdirs, files in os.walk(indir):
            # checking if in the correct directory
            if 'output.txt' not in files:
                continue

            data_file_name = os.path.join(dirs, 'output.txt')
            w0_footprint = []
            wn_footprint = []
            for line in open(data_file_name, 'r'):
                match = regexp1.search(line)
                if not match:
                    continue
                wid, footprint = match.groups()
                wid = int(wid)
                footprint = float(footprint)
                if wid == 0:
                    w0_footprint.append(footprint)
                else:
                    wn_footprint.append(footprint)
            if len(w0_footprint) != 1:
                print(f'Error with file: {data_file_name}')
                continue
            # w0_footprint = w0_footprint[0]
            exe = data_file_name.split('/')[-5]
            mem = data_file_name.split('_mem')[1].split('_')[0]
            gzip = data_file_name.split('_gzip')[1].split('_')[0]
            chunk = data_file_name.split('_chunk')[1].split('_')[0]
            wpn = data_file_name.split('_wpn')[1].split('_')[0]
            N = data_file_name.split('_N')[1].split('_')[0]
            mpi = data_file_name.split('_mpi')[1].split('_')[0]
            key = f'{exe}-{mem}-{gzip}-{chunk}-{N}-{wpn}-{mpi}'
            if wn_footprint:
                value = [w0_footprint[0], np.mean(wn_footprint), np.std(wn_footprint)]
            else:
                value = [w0_footprint[0], 0, 0]
            if key in dic:
                dic[key].append(value)
            else:
                dic[key] = [value]
        rows = []
        for key, val in dic.items():
            val = np.mean(val, axis=0).round(3)
            row = key.split('-') + list(val.astype(str))
            rows.append(row)
        rows.sort(key=lambda a: (a[0], float(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5])))

        with open(f'{indir}/results.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            writer.writerow(header)
            writer.writerows(rows)

    print('\nCompleted Successfully!')