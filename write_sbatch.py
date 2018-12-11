#!/usr/bin/python

NN=14
N=33
# generic content
beginning = u"""#!/bin/bash
#SBATCH -o /naslx/projects/pr63so/ga36wom2/gauss-newton-hessian-optimizer/myjob.%j.%N.out 
#SBATCH -D /naslx/projects/pr63so/ga36wom2/gauss-newton-hessian-optimizer
#SBATCH -J small-cnn
#SBATCH --get-user-env 
#SBATCH --clusters=mpp3
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=128
#SBATCH --mail-type=end 
#SBATCH --mail-user=ga36wom@mytum.de 
#SBATCH --export=NONE 
#SBATCH --time=01:30:00 
module load python
source activate sev_python
# source /etc/profile.d/modules.sh
cd /naslx/projects/pr63so/ga36wom2/gauss-newton-hessian-optimizer
export OMP_NUM_THREADS=128
# 256 is the maximum reasonable value for CooLMUC-3
export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
"""



for n in range(N):
	content = beginning
	content += u"""\nexport NN="""+str(NN)
	content += u"""\nexport node_number="""+str(n)  # must be [0 .. nodes-1]
	content += u"""\nexport nodes="""+str(N)
	
	
	

	content += u"""\nfor i in `seq 1 $NN`; \ndo numactl -m 0 python cnn.py $i $NN $node_number $nodes &"""
	content += u"""echo "i and NN is" $i $NN \ndone \nwait"""
	
	text_file = open ("sbatch_python_writen"+format(n,'03d')+".sh","w")
	text_file.write(content)
	text_file.close()
