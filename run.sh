<<<<<<< Updated upstream
LQDIR=$1
OUTDIR=$2
REFDIR=$3

python ~/College/24/zyan/RestoreFormerPlusPlus/inference.py -i $LQDIR -o $OUTDIR -v RestoreFormer++ --save
=======
python inference.py -i ../meLQ/hasGT/64 -o results/RF++/meResults -v RestoreFormer++ -s 2 --save
#python inference.py -i ../meLQ/hasGT/32 --ref_dir ../meHQ -o results/RF++/meResults32 -v RestoreFormer++ -s 2 --save
#python inference.py -i ../meLQ/hasGT/32 --ref_dir ../meHQ -o results/RF++/meResults32NoQuant -v RestoreFormer++ -s 2 --save
#python inference.py -i data/pgLR -o results/RF++/pgLR -v RestoreFormer++ -s 2 --save
>>>>>>> Stashed changes
