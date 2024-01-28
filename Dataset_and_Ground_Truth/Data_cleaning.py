from library import *
from functions import *
import csv
import ast
path="Dataset_and_Ground_Truth/tsv/Marianne_0024.tsv"
path1="Dataset_and_Ground_Truth/tsv/Marianne_0024_prova.tsv"

#funtions for dealing with files.
def tsv_read(path):
    dati = []
    with open(path, 'r', newline='', encoding='utf-8') as file_tsv:
        lettore_tsv = csv.DictReader(file_tsv, delimiter='\t')
        for riga in lettore_tsv:
            # Converti le stringhe contenenti rappresentazioni di liste in liste Python
            for chiave, valore in riga.items():
                try:
                    riga[chiave] = ast.literal_eval(valore)
                except (SyntaxError, ValueError):
                    pass
            dati.append(dict(riga))
    return dati

def tsv_write(new_path, dati, original_fieldnames):
    if not dati:
        print("Errore: La lista dei dati è vuota.")
        return

    # Utilizza l'ordine originale delle chiavi
    fieldnames = original_fieldnames

    with open(new_path, 'w', newline='', encoding='utf-8') as file_tsv:
        scrittore_tsv = csv.DictWriter(file_tsv, fieldnames=fieldnames, delimiter='\t')
        scrittore_tsv.writeheader()

        # Scrivi i dati nel file TSV, convertendo le liste in stringhe
        for riga in dati:
            riga_con_stringhe = {k: str(v) if not isinstance(v, list) else str(v).replace(',', '') for k, v in riga.items()}
            scrittore_tsv.writerow(riga_con_stringhe)
'''''''''
def tsv_read(path):
    dati = []
    with open(path, 'r', newline='', encoding='utf-8') as file_tsv:
        lettore_tsv = csv.DictReader(file_tsv, delimiter='\t')
        for riga in lettore_tsv:
            dati.append(dict(riga))
    return dati
def tsv_write(path, dati):
    with open(path, 'w', newline='', encoding='utf-8') as file_tsv:
        fieldnames = set().union(*(d.keys() for d in dati))
        scrittore_tsv = csv.DictWriter(file_tsv, fieldnames=fieldnames, delimiter='\t')
        scrittore_tsv.writeheader()
        scrittore_tsv.writerows(dati)
###
'''''

dati_del_file = tsv_read(path)
print(len(dati_del_file))

ll = [[str(dizionario[chiave]) for chiave in dizionario] for dizionario in dati_del_file]
v=np.array(ll)
v=v.astype(str)
for i in range(0,v.shape[0]):
    dec=v[i,1].split(".")
    if len(dec[1]) ==1:
        v[i,1]+="0000"
    else:
        v[i,1]+="000"
print(v[:,1])

'''''''''
v=np.array(dati_del_file[0]).reshape((len(dati_del_file[0]),1)).T

for righe in range(0,len(dati_del_file)):
    if  righe> 10:
        v=np.vstack((v,np.array(dati_del_file[righe]).reshape((dati_del_file[righe],1)).T))
'''''
data=np.array(v).astype(str)
for j in range(0,v.shape[1]):
    if v[v.shape[0]-1,j].astype(str)=="0.000" or v[v.shape[0]-1,j].astype(str)=="NAN" or v[v.shape[0]-1,j].astype(str)=="0.0":
        k=v.shape[0]-1
        
        while v[k,j].astype(str)=="0.000" or v[k,j].astype(str)=="NAN" or v[k,j].astype(str)=="0.0":
            k=k-1
        data[v.shape[0]-1,j]=v[k,j].astype(str)
  
for j in range(0,data.shape[1]):
    if j>2:
        for i in range(0,data.shape[0]):
            if(data[i,j].astype(str)=="NAN" or data[i,j].astype(str)=="0.000" or data[i,j].astype(str)=="0.0"):
                lower=data[i-1,j].astype(float)
                #upper=lower
                fN=i
                while data[i,j].astype(str)=="NAN" or data[i,j].astype(str)=="0.000" or data[i,j].astype(str)=="0.0":
                    i=i+1
                lN=i-1
                upper=data[i,j].astype(float)
                d1=lN-fN+1
                d2=upper-lower
                incr=d2/(d1+1)
                
                for z in range (fN,i):
                        data[z,j]=(np.round(lower+incr*(z-fN+1),decimals=3)).astype(str)
                

'''''
for j in range(0,data.shape[1]):
    indx=np.where(data[:,j]=="NAN")
    if np.any(data[:,j]=="NAN"):
 '''       
for i in range(0,len(dati_del_file)):
    j=0
    for chiave in dati_del_file[0]:
        dati_del_file[i][chiave]=data[i,j]
        j=j+1
    
original_fieldnames = dati_del_file[0].keys()

# Chiamata a tsv_write con l'ordine originale delle chiavi
tsv_write(path1, dati_del_file, original_fieldnames)