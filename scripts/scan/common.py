import os
import itertools
import numpy as np
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

home = os.path.abspath(os.path.join(this_directory, '../../'))
exe_home = os.path.join(home, '')
# batch_script = os.path.join(blond_home, 'scripts/other/batch-simple.sh')

mpirun = 'mpirun'
python = 'python'

cores_per_cpu = 20

def get_permutations(my_dict):
    #keys, values = zip(*my_dict.items())
    #permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    all_names = list(my_dict.keys())
    permutations = list(itertools.product(*(my_dict[Name] for Name in all_names)))
    return permutations


def get_unique_combinations(my_dict):
    maxlen = np.max([len(v) if isinstance(v, list)
                     else 1 for k, v in my_dict.items()])
    for k, v in my_dict.items():
        if isinstance(v, list):
            assert maxlen % len(v) == 0, 'Size of {} must be a multiple of {}'.format(len(v), maxlen)
            my_dict[k] = v * int(maxlen / len(v))
        else:
            my_dict[k] = [v] * maxlen
        assert len(my_dict[k]) == maxlen
    temp_configs = list(zip(*my_dict))
    return temp_configs

# Evolve SLURM flags
evolve = {
    'script': os.path.join(home, 'scripts/other/evolve-slurm-simple.sh'),
    'submit': 'sbatch',
    'run': 'srun',
    'nodes': '--nodes',
    'workers': '--ntasks',
    'tasks_per_node': '--ntasks-per-node',
    'cores': '--cpus-per-task',
    'time': '-t',
    'output': '-o',
    'error': '-e',
    'jobname': '-J',
    'partition': '--partition',
    # 'gpu': '--gres=gpu:',
    'default_args': [
            '--mem', '0',
            '--export', 'ALL',
            '--hint', 'nomultithread',
            '--gres', 'gpu:2'
            # '--overcommit'
            # '--partition', 'inf-short'
    ]
}



# SLURM flags
slurm = {
    'script': os.path.join(home, 'scripts/other/batch-simple.sh'),
    'submit': 'sbatch',
    'run': 'srun',
    'nodes': '--nodes',
    'workers': '--ntasks',
    'tasks_per_node': '--ntasks-per-node',
    'cores': '--cpus-per-task',
    'time': '-t',
    'output': '-o',
    'error': '-e',
    'jobname': '-J',
    'partition': '--partition',

    'default_args': [
            '--mem', '0',
            '--export', 'ALL',
            # '--overcommit'
            '--hint', 'nomultithread',
            # '--partition', 'inf-short'
    ]
}

# HTCondor args
condor = {
    'submit': 'condor_submit',
    'script': os.path.join(home, 'scripts/other/condor.sub'),
    'executable': 'executable='+os.path.join(home, 'scripts/other/condor.sh'),
    'output': 'output=',
    'error': 'error=',
    'log': 'log=',
    'arguments': 'arguments=',
    'jobname': '-batch-name',
    'time': '+MaxRuntime=',
    'cores': 'request_cpus=',
    'gpus': 'request_GPUs=',
    'default_args': [
        # 'requirements=regexp("V100", TARGET.CUDADeviceName)'
        # 'environment="PATH={PATH} PYTHONPATH={PYTHONPATH}"',
        # 'getenv=True',
        # 'should_transfer_files=IF_NEEDED'
    ]

}
