import subprocess
import os
import sys
from datetime import datetime
import random
import yaml
import argparse
import numpy as np
from time import sleep
import common

job_name_form = '_mem{memory}_gzip{gzip}_chunk{chunk}_wpn{wpn}_N{nodes}_mpi{mpi}_'

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Run MPI experiments.',
                                 usage='python {} -t numpy h5py'.format(this_filename[:-3]))

parser.add_argument('-e', '--environment', type=str, default='slurm', choices=['slurm'],
                    help='The environment to run the scan.')

parser.add_argument('-t', '--testcases', type=str, nargs='+', choices=['h5py', 'numpy'],
                    help='Which testcases to run. Default: all')

parser.add_argument('-o', '--output', type=str, default='./results/raw',
                    help='Output directory to store the output data. Default: ./results/raw')

parser.add_argument('-l', '--limit', type=int, default=0,
                    help='Limit the number of concurrent jobs queueing. Default: 0 (No Limit)')

if __name__ == '__main__':
    args = parser.parse_args()
    top_result_dir = args.output
    # os.environ['HOME'] = common.home
    # os.environ['FFTWDIR'] = os.environ.get('FFTWDIR', '$HOME/install')
    # os.environ['HOME'] =
    for tc in args.testcases:
        yc = yaml.load(open(this_directory + '/{}_configs.yml'.format(tc), 'r'),
                       Loader=yaml.FullLoader)[args.environment]

        result_dir = top_result_dir + '/{tc}/{config_name}/{job_name}/{timestr}/{fname}'

        total_sims = 0
        # Calculate all configurations for every run_config
        # Store them in the all_configs dictionary
        all_configs = {}
        for rc in yc['run_configs']:
            rc_d = yc['configs'][rc]
            comb_type = rc_d.get('combinations', ['unique'])[0]
            temp_configs = []
            if comb_type == 'unique':
                temp_configs = common.get_unique_combinations(rc_d)
            elif comb_type == 'cross':
                temp_configs = common.get_permutations(rc_d)
            else:
                print(f'[Warning] Incorrect combination style for {tc}:{rc}')
            total_sims += len(temp_configs)
            all_configs[rc] = temp_configs.copy()

        print("Total runs: ", total_sims)

        current_sim = 0
        for config_name, configs in all_configs.items():
            keys = list(yc['configs'][config_name])
            for config in configs:
                config = dict(zip(keys, config))
                job_name = job_name_form.format(**config)
                #     config[keys.index('memory')],
                #     config[keys.index('gzip')],
                #     config[keys.index('chunk')],
                #     config[keys.index('wpn')],
                #     config[keys.index('nodes')],
                #     config[keys.index('mpi')]
                # )

                for i in range(int(config['repeats'])):
                    timestr = datetime.now().strftime('%d%b%y.%H-%M-%S')
                    timestr = timestr + '-' + str(random.randint(0, 100))
                    output = result_dir.format(tc=tc, config_name=config_name, job_name=job_name,
                                               timestr=timestr, fname='output.txt')
                    error = result_dir.format(tc=tc, config_name=config_name, job_name=job_name,
                                              timestr=timestr, fname='error.txt')
                    if not os.path.exists(os.path.dirname(output)):
                        os.makedirs(os.path.dirname(output))
                    # tc, config_name, job_name, timestr, 'output.txt')
                    # error = result_dir.format(
                    #     tc, config_name, job_name, timestr, 'error.txt')
                    # condor_log = result_dir.format(
                    #     tc, config_name, job_name, timestr, 'log.txt')
                    # monitorfile = result_dir.format(
                    #     tc, config_name, job_name, timestr, 'monitor')
                    # log_dir = result_dir.format(
                    #     tc, config_name, job_name, timestr, 'log')
                    # report_dir = result_dir.format(
                    #     tc, config_name, job_name, timestr, 'report')
                    # for d in [log_dir, report_dir]:
                    #     if not os.path.exists(d):
                    #         os.makedirs(d)

                    os.environ['OMP_NUM_THREADS'] = str(config['omp'])

                    analysis_file = open(os.path.join(top_result_dir, tc,
                                                      config_name, '.analysis'), 'a')

                    exe_args = [common.python]
                    # os.path.join(common.exe_home, exe)]
                    for argname, argvalue in config.items():
                        if argname == 'exe':
                            exe_args.append(os.path.join(common.exe_home, argvalue))
                        elif argname in ['memory', 'gzip', 'chunk']:
                            exe_args.append(f'--{argname}={argvalue}')

                    if args.environment in ['slurm', 'cloud']:
                        batch_args = [
                            common.slurm['submit'],
                            common.slurm['nodes'], str(config['nodes']),
                            # common.slurm['workers'], str(w),
                            common.slurm['tasks_per_node'], str(config['wpn']),
                            common.slurm['cores'], str(config['omp']),  # str(o),
                            common.slurm['time'], str(config['time']),
                            common.slurm['output'], output,
                            common.slurm['error'], error,
                            common.slurm['jobname'], tc + '-' + config_name + job_name.split('/')[0] + '-' + str(i),
                            common.slurm['partition'], str(config['partition'])]
                        batch_args += common.slurm['default_args']
                        batch_args += [common.slurm['script'],
                                       common.slurm['run']]
                        all_args = batch_args + exe_args
                    else:
                        print('[Error] Wrong environment!')
                        sys.exit(-1)

                    print(job_name, timestr)
                    print(job_name, timestr, "\n", file=analysis_file)

                    all_args = ' '.join(all_args)
                    print(all_args, "\n", file=analysis_file)

                    if args.limit > 0 and args.environment == 'slurm':
                        # Calculate the number of jobs currently running
                        jobs = subprocess.run(
                            'squeue -u $USER | wc -l', shell=True,
                            stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                        jobs = int(jobs) - 1
                        # While the number of jobs in the queue are more
                        # or equal to the jobs limit, wait for a minute and repeat
                        while jobs >= args.limit:
                            sleep(60)
                            jobs = subprocess.run(
                                'squeue -u $USER | wc -l', shell=True,
                                stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                            jobs = int(jobs) - 1

                    subprocess.call(all_args,
                                    shell=True,
                                    stdout=open(output, 'w'),
                                    stderr=open(error, 'w'),
                                    env=os.environ.copy())

                    current_sim += 1
                    print(f"{current_sim}/{total_sims} ({100.0 * current_sim / total_sims}) has been completed")

                    analysis_file.close()
