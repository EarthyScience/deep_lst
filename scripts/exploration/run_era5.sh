#!/bin/bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=scripts/out/era5_%j
#SBATCH --job-name=era5_skt_lst

# ERA5-Land SKT vs SEVIRI LST similarity over Africa: cubes on work_5, then the PDFs.
# Submit from the project root:
#   sbatch scripts/exploration/run_era5.sh              # skt_lst -> era5/
#   sbatch scripts/exploration/run_era5.sh swvl1_lst    # other pairs -> era5/<pair>/

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"
# venv GDAL bindings need the newer libgcc_s / libstdc++ (same line as in .bashrc)
export LD_LIBRARY_PATH=$HOME/.local/lib/gdal-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

PAIR=${1:-skt_lst}
CUBE_DIR=/Net/Groups/BGI/work_5/scratch/mwessel/results/independent/era5
FIG_DIR=${WORKDIR}/results/independent/era5
if [ "$PAIR" != "skt_lst" ]; then
    CUBE_DIR=${CUBE_DIR}/${PAIR}
    FIG_DIR=${FIG_DIR}/${PAIR}
fi

python3 -u ${WORKDIR}/deep_lst/exploration/era5.py compute --pair ${PAIR} --workers 8 --cube_dir ${CUBE_DIR}
python3 -u ${WORKDIR}/deep_lst/exploration/era5.py plot --cube_dir ${CUBE_DIR} --fig_dir ${FIG_DIR}
