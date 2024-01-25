# Dataset and Ground Truth

## data folder
On data folder you'll find the output of EyesWeb. each file has 8 columns:
1. Global KE
2. Chest KE
3. Head KE
4. Point Density
5. Left wrist KE
6. Right wrist KE
7. Left ankle KE
8. Right ankle KE
(KE= Kinetic Energy)

for our first pourpuse we had not considered the Chest KE (2) 

## Ground_Truth folder
There you can find for each video in which second (with centesimal precision, according with ELAN elaborations) occurs a saliency for us. (the saliency is oriented to the ML algorithm)

## tsv folder
In this folder you will find all the tsv cleaned (by using a linear interpolation) from NAN values. 
Further, you can find in the TSV_OLD.zip all the tsv which are not cleaned.

P.S. we could have to adjust some stuff on the algorithm, then maybe we will add some tsv or modify them before using them on the prediction of saliency (ML algorithm).
