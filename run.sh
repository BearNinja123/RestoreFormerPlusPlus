LQDIR=$1
OUTDIR=$2
REFDIR=$3

python ~/College/24/zyan/RestoreFormerPlusPlus/inference.py -i $LQDIR -o $OUTDIR -v RestoreFormer++ --save
