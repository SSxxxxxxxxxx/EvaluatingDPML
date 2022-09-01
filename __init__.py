import sys, os,inspect

# This allows one to work with the repository as a submodule
print('Loding submodule evaluatingDMPL')

# Allows uncomplicated access to the submodule-folder
evaluatingDPML_path = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))

# Allows importing within the submodule
if evaluatingDPML_path not in sys.path:
    sys.path.append(evaluatingDPML_path)


# Some scripts need specific working directory
def chdir_to_evaluating():
    os.chdir(os.path.join(evaluatingDPML_path, 'evaluating_dpml'))


def chdir_to_improved_mi():
    os.chdir(os.path.join(evaluatingDPML_path, 'improved_mi'))


def chdir_to_dataset():
    os.chdir(os.path.join(evaluatingDPML_path, 'dataset'))
