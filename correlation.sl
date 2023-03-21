#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 05:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name density_correlation
#SBATCH --output density_correlation.out
#SBATCH --error density_correlation.err

module load python
source activate picca_dev
umask 0002
export OMP_NUM_THREADS=32

deltas_dir=/pscratch/sd/c/cramirez/deltas_from_colore/deltas
quasar_catalog=/pscratch/sd/c/cramirez/deltas_from_colore/zcat.fits
output_file=/pscratch/sd/c/cramirez/deltas_from_colore/xcf.fits.gz

command="picca_xcf.py --in-dir ${deltas_dir} --drq ${quasar_catalog} --out ${output_file} --nproc 32 --mode desi_mocks --no-project --no-remove-mean-lambda-obs"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
