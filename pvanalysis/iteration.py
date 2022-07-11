import subprocess
import shlex

for i in range(100):
    print(i)
    subprocess.run(shlex.split('python /Users/yusukeaso/C_programs/Programs/SLAM/pvanalysis/SpM.py'))
    subprocess.run(shlex.split('python /Users/yusukeaso/C_programs/Programs/SLAM/pvanalysis/clean.py'))

