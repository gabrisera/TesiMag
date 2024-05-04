from library import *
from functions import *
from dataset import *
## remove all the firsts rows on zeros (labled to 1)
samples=joblib.load("saved_var/samples_2.pkl")
path1="samples/Cora/"
path2="samples/Marianne/"
path3="samples/Muriel/"
i=0
#"0004_2019-05-22","0005"
cora=["0001","0004_2019-05-22","0004_2019-08-08","0005","0014"]
marianne=["0007","0008","0010","0018","0019","0024","0026","0041","0042","0043","0047","0048"]
muriel=["0018","0026","0027","0030"]
for el in samples["Cora"]:
    
    path=path1+"Cora_"+cora[i]+".txt"
    print(path)
    matrix=np.array(el)
    flag=False
    flag2=False
    
    for j in range(0,matrix.shape[0]):
        if matrix[j,matrix.shape[1]-1]!=2:
             flag2=True
        for z in range(0,matrix.shape[1]-1):
            if matrix[j,z]!=0:
                k=j
                flag=True
                break
        if  flag and flag2:
            break
    
    matrix=matrix[k:,:]
    
    for m in range(0,matrix.shape[1]-1):
        matrix[:,m]=normalize(matrix[:,m])
    
    np.savetxt(path, matrix, fmt='%f', delimiter=' ')
    samples["Cora"][i]=matrix
    i=i+1
i=0
for el in samples["Marianne"]:
    path=path2+"Marianne_"+marianne[i]+".txt"
    print(path)
    matrix=np.array(el)
    
    flag=False
    flag2=False
    for j in range(0,matrix.shape[0]):
        if matrix[j,matrix.shape[1]-1]!=2:
             flag2=True
        for z in range(0,matrix.shape[1]-1):
            if matrix[j,z]!=0:
                k=j
                flag=True
                break
        if  flag and flag2:
            break
    matrix=matrix[k:,:]
    
    for m in range(0,matrix.shape[1]-1):
        matrix[:,m]=normalize(matrix[:,m])
    
    np.savetxt(path, matrix, fmt='%f', delimiter=' ')
    samples["Marianne"][i]=matrix
    i=i+1
i=0
for el in samples["Muriel"]:
    path=path3+"Muriel_"+muriel[i]+".txt"
    print(path)
    matrix=np.array(el)
    flag=False
    flag2=False
    for j in range(0,matrix.shape[0]):
        if matrix[j,matrix.shape[1]-1]!=2:
             flag2=True
        for z in range(0,matrix.shape[1]-1):
            if matrix[j,z]!=0:
                k=j
                flag=True
                break
        if  flag and flag2:
            break
    matrix=matrix[k:,:]
    
    for m in range(0,matrix.shape[1]-1):
        matrix[:,m]=normalize(matrix[:,m])
    
    np.savetxt(path, matrix, fmt='%f', delimiter=' ')
    samples["Muriel"][i]=matrix
    i=i+1
joblib.dump(samples,"saved_var/samples_2.pkl")
'''
flag=True
    for j in range(0,matrix.shape[0]):
        for z in range(0,matrix.shape[1]-1):
            if matrix[j,z]!=0:
                k=j
                flag=False
                break
        if not flag:
            break
            '''