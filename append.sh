#!/bin/bash
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
#SBATCH --time=00:10:00 
module load python
source activate sev_python
# source /etc/profile.d/modules.sh
cd /naslx/projects/pr63so/ga36wom2/gauss-newton-hessian-optimizer
export OMP_NUM_THREADS=128
# 256 is the maximum reasonable value for CooLMUC-3
export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

export NN=14
export node_number=30
export nodes=33

numactl -m 0 python append-files.py &
# numactl -m 0 python cnn.py 9 $NN $node_number $nodes &
# for i in `seq 1 $NN`; 
# do numactl -m 0 python cnn.py $i $NN $node_number $nodes &echo "i and NN is" $i $NN 
# done 
wait
