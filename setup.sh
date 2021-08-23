#!/bin/bash
#SBATCH --mem=5g
#SBATCH -c1
#SBATCH --time=3-0
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=gal.patel@mail.huji.ac.il
#SBATCH --wckey=strmt
#SBATCH --killable

dir=/cs/labs/oabend/gal.patel/projects/MT_eval
cd $dir
source /cs/labs/oabend/gal.patel/virtualenvs/mteval-venv/bin/activate
python evaluate_utils.py