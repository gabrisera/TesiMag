from functions import *
from library import *
from dataset import *
'''''
x=np.array([4,10,33,4,44,4,4,-20,4,12,58,4,4,4,2,-47,4,44,4,21,3])
x1=np.array([10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,
             10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80])

x2 = np.random.uniform(10,40,304)
ripetit=rip_index(x1,x2)
print(f"mag e freq{ripetit}")
'''
df=import_data1()
valori_del_dizionario = list(df.values())

# Ottieni il numero di righe e colonne
numero_di_righe = len(valori_del_dizionario[0])
numero_di_colonne = len(valori_del_dizionario[0]) if valori_del_dizionario else 0

matrix=np.empty((0,8))
for chiave, valore in df.items():
    if not matrix.size:
        matrix=np.array(valore.values)
        
    else:
        matrix=np.vstack((matrix,valore.values))

# Stampa le dimensioni dell'elemento

matrix=matrix[::2]
print(matrix.shape)
density=normalize(matrix[:,3]-400)
k_e_l_w=normalize(matrix[:,4])
k_e_r_w=normalize(matrix[:,5])
k_e_l_a=normalize(matrix[:,6])
k_e_r_a=normalize(matrix[:,7])
matrix1=np.column_stack((density,k_e_l_w,k_e_r_w,k_e_l_a,k_e_r_a))
col=[0,1,2,3,4]
matrix1[:,col]=np.where(matrix1[:,col]<0.07,0,matrix1[:,col])
percorso_file="prova.txt"
np.savetxt(percorso_file, matrix1, delimiter='\t', fmt='%.4f')
print(matrix1.shape)
#ricordarsi che è indice +hk il frame corrispondente nel video
k=51
hk=int(np.floor(k/2))
windows=sliding_windows1(matrix1,k).T
#windows=windows[hk:,:]
print(windows.shape)
percorso_file = 'matrice_output1.txt'
col=[0,2,4,6,8]
windows[:, col] = np.where(windows[:, col] > 0.005, 1, windows[:, col])
windows[0:hk,col]=1
np.savetxt(percorso_file, windows, delimiter='\t', fmt='%.4f')
#indici=np.where(windows[:,2]<0.005 or windows[:,4]<0.005 or windows[:,6]<0.005 or windows[:,8]<0.005)


indici1 = np.where(windows[:, col[0]] < 0.005)
indici2 = np.where(windows[:, col[1]] < 0.005)
indici3 = np.where(windows[:, col[2]] < 0.005)
indici4 = np.where(windows[:, col[3]] < 0.005)
indici5 = np.where(windows[:, col[4]] < 0.005)

# Funzione per applicare la tolleranza a tutti gli indici


indici_and_1_2 = intersec(indici1, indici2)
indici_and_1_3 = intersec(indici1, indici3)
indici_and_1_4 = intersec(indici1, indici4)
indici_and_1_5 = intersec(indici1, indici5)
indici_and_2_3 = intersec(indici2, indici3)
indici_and_2_4 = intersec(indici2, indici4)
indici_and_2_5 = intersec(indici2, indici5)

indici_and_3_5 = intersec(indici3, indici5)
indici_and_4_5 = intersec(indici4, indici5)
indici_and_3_4 = intersec(indici3, indici4)
#condizione = np.logical_or.reduce([windows[:, 2] < 0.005, windows[:, 4] < 0.005, windows[:, 6] < 0.005, windows[:, 8] < 0.005])
#indici = np.where(condizione)
#print(len(indici))
#indici=np.unique(indici[0])

print(f"indici_and_3_4{indici1}")

