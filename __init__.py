import sys, os
print('Adding evaluationDPML to PATH', )
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'evaluating_dpml'))