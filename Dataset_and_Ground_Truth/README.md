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
There you can find for each video in which second (with centesimal precision, according with ELAN elaborations) occurs a saliency for us. 

*The saliency is oriented to the ML algorithm*.

## tsv folder
In this folder you will find all the tsv (EyesWeb input) about markers position *cleaned* from NAN values. 

Further, you can find in the TSV_OLD.zip all the not cleaned tsv.

## Data_cleaning.py

This python file is the algorithm for clening the tsv from nan values. 

*if you use it, remind to delete the intestation of tsv (11 rows), it will give you output a file with data cleaned in which you have to add the intestation that you have deleted before (without intestation EyesWeb doesnt work, and with intestation the file Data_cleaning.py doesnt work).*
