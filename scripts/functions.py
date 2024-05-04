from library import *

#funzione per normalizzare i dati 
def normalize(vettore):
    """
    """
    minimo = np.min(vettore)
    massimo = np.max(vettore)
    vettore_normalizzato = (vettore - minimo) / (massimo - minimo)
    return vettore_normalizzato

#funzione per decidere quali video finiscono per il testing e quali per il training 
def split_data(df,df1):
    Xt = np.empty((0, 4))  
    Yt = np.empty((0, 1))  

    Xl = np.empty((0, 4))
    Yl = np.empty((0, 1))

    for chiave, valore in df.items():
        if (chiave == "23-03-2016_t030" or chiave == "22-03-2016_t048" or chiave == "21-03-2016_t027" or chiave == "08-08-2019_t020"):
            if not Xt.size:  # Se Xt è vuoto, assegna direttamente la matrice
                Xt = np.array(valore.values)
                Yt = np.array(df1[chiave].values)
            else:  # Altrimenti, verifica le dimensioni prima di concatenare
                if Xt.shape[1] == valore.shape[1]:
                    Xt = np.vstack((Xt, valore.values))
                    Yt = np.vstack((Yt, df1[chiave].values))
                else:
                    print(f"Le dimensioni delle colonne di {chiave} non coincidono.")
        else:
            if not Xl.size:  # Se Xl è vuoto, assegna direttamente la matrice
                Xl = np.array(valore.values)
                Yl = np.array(df1[chiave].values)
            else:  # Altrimenti, verifica le dimensioni prima di concatenare
                if Xl.shape[1] == valore.shape[1]:
                    Xl = np.vstack((Xl, valore.values))
                    Yl = np.vstack((Yl, df1[chiave].values))
                else:
                    print(f"Le dimensioni delle colonne di {chiave} non coincidono.")
    return Xl,Yl,Xt,Yt

#funzione per aggiungere dei valori aggiuntivi alla label 
def add_values_c(Yl):
    for i in range(1, np.size(Yl)):
        if Yl[i] == 1 and Yl[i - 1] == 0:
            for j in range(9, 0, -1):
                if i + j + 1 < Yl.size:
                    if [i + abs(10-j)]==1 or [i - abs(10-j)]==1:
                        break
                    else:
                        Yl[i + abs(10-j)] = 1
                        Yl[i - abs(10-j)] = 1
                        #print(i + abs(5-j))
                else:
                    break

    return Yl
def add_values_r(Yl):
    for i in range(1, np.size(Yl)):
        if Yl[i] == 1:
            Yl[i]=100
            Yl[i-1]=100
            Yl[i-2]=100
            if all(Yl[i-1:i-9] == 0) and all(Yl[i+1:i+9] == 0) and i>10:
                for j in range(9, 1, -1):
                    if i + j < Yl.size and i - j >= 0:
                        Yl[i+10-j] = j
                        Yl[i-12+j] = j
                    else:
                        break
            else:
                continue
        else:
            continue
    return Yl
#funzione per stampare quanti e quali valori sono presenti in un vettore
def print_val_count(x):
    unique, count=np.unique(x,return_counts="true")
    x_val_count={ k:v for (k,v) in zip(unique,count) }
    print(f'conta valori: {x_val_count}')

#funzione per creare le sliding windows    
def sliding_windows(x,k):

    ns=x.size
    hk=math.floor(k/2) 
    win=np.zeros((7,ns-hk))
    #np.shape(win)
   
    print(hk)
    for i in range(hk,ns):
        stat=[]
        #print(hk,ns)
        if i+hk>=ns:
            break
        
        w=np.transpose(np.vstack([x[i-hk:i],x[i+1:i+hk]]))  
        rip_i=rep_index(w[:,:hk],w[:,hk:])    
        media1=np.mean(w[:,:hk])
        media2=np.mean(w[:,hk:])
        media=np.abs(media2-media1)      
        varianza1=np.var(w[:,:hk])
        varianza2=np.var(w[:,hk:])
        varianza=np.abs(varianza2-varianza1)
        moda1=np.argmax(stats.mode(w[:,:hk]).count)
        moda2=np.argmax(stats.mode(w[:,hk:]).count)
        moda=np.abs(moda2-moda1)
        median1=np.median(w[:,:hk])
        median2=np.median(w[:,hk:])
        median=np.abs(median2-median1)
        std1=np.std(w[:,:hk])
        std2=np.std(w[:,hk:])
        std=np.abs(std2-std1)
        int1=np.std(w[:,:hk])
        int2=np.std(w[:,hk:])
        intervallo=np.abs(int2-int1)
        #corr_coeff=np.corrcoef(w[:,:hk],w[:,hk:])[0,1]
        '''if np.isnan(corr_coeff):
            corr_coeff=0
            '''
        stat.append([media,varianza,median,moda,std,intervallo,rip_i])
        #print(media.shape,varianza.shape,median.shape,moda.shape,std.shape,intervallo.shape,corr_coeff.shape)
        #print(len(stat[0]),len(stat[1]),len(stat[2]),len(stat[3]),len(stat[4]),len(stat[5]),len(stat[6]))
        #print(st.shape)
        st=np.array(stat)
        #w=np.append(w,stat)
        #print(i)
        win[:,i]=st.reshape(-1)
        #print(win[:,i])
        del stat
        #w[:]
    
    return win

#funzione per split training e validation
def split_train_val(Q, Yl, train_ratio, val_ratio):
    
    
    print(f"Q {Q.shape} Yl {Yl.shape}")
    # Calcolo le dimensioni per ciascun set
    num_rows = Q.shape[0]
    train_size = int(num_rows * train_ratio)
    val_size = num_rows-train_size
    #test_size = num_rows - train_size - val_size

    # Genero gli indici per gli insiemi
    indices = np.arange(num_rows)
    np.random.shuffle(indices)

    # Divido gli indici nei vari set
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Creo i set utilizzando gli indici
    x_train = Q[train_indices, :]
    x_val = Q[val_indices, :]


    #divido le lablel in base agli indici
    Y_train=Yl[train_indices]
    Y_val=Yl[val_indices]
    return x_train, x_val, Y_train, Y_val

#funzione per downsampling
def downsampling(x_train,y_train,flag):
    df_train = pd.concat([pd.DataFrame(x_train), pd.Series(y_train.reshape(-1), name='target_column')], axis=1)

    # Separare le classi maggioritarie e minoritarie
    df_majority = df_train[df_train['target_column'] == 0]  # sostituisci 'target_column' con il tuo nome di colonna target
    if(not flag):
        df_minority = df_train[df_train['target_column'] == 1]
    else:
        df_minority = df_train[df_train['target_column'] == 100]
        df_minority1 = df_train[df_train['target_column'] == 9]
        df_minority2 = df_train[df_train['target_column'] == 8]
        df_minority3 = df_train[df_train['target_column'] == 7]
        df_minority4 = df_train[df_train['target_column'] == 6]
        df_minority5 = df_train[df_train['target_column'] == 5]
        df_minority6 = df_train[df_train['target_column'] == 4]
        df_minority7 = df_train[df_train['target_column'] == 3]
        df_minority8 = df_train[df_train['target_column'] == 2]
        
    # Downsampling delle classi maggioritarie
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=int(4*len(df_minority)), random_state=42)
    if not flag:
        df_downsampled=pd.concat([df_majority_downsampled,df_minority])
    else:
    # Combinare i campioni downsampled con la classe minoritaria
        df_downsampled = pd.concat([df_majority_downsampled, df_minority, df_minority1,
                                    df_minority2, df_minority3, df_minority4, df_minority5, 
                                    df_minority6, df_minority7, df_minority8,])

    # Agitare il DataFrame risultante
    df_downsampled = df_downsampled.sample(frac=1, random_state=42)

    # Separare nuovamente X_train e y_train
    X_train_downsampled = df_downsampled.drop('target_column', axis=1)
    y_train_downsampled = df_downsampled['target_column']
    return X_train_downsampled.values, y_train_downsampled.values

#feature ripetiviness prima parte dell'array rispetto seconda parte
def rep_index(x,y):
    f_diff=0
    rip1,f1,reg1=mean_occ_fur(x)
    rip2,f2,reg2=mean_occ_fur(y)
    
    ''''
    if index==2014:
        rip1=mean_occ_fur(x,True)
        rip2=mean_occ_fur(y,True)
        print(f"rip1, rip2 {rip1,rip2}")
        '''
    #print(f"rip1 rip2 {rip1,rip2}")
    #print(f1,f2)
    if(rip1>rip2):
        coeff=rip2/rip1
    else:
        coeff=rip1/rip2
    if coeff>0.5 and reg1==1 and reg2==1:
        f_diff=np.abs(f1-f2)
    return coeff, f_diff
#calcolo valore repitiveness
def mean_occ_fur(x):
    n = x.shape[0]
    
    
    '''''
    # Zero-padding (opzionale)
    padded_size = 2**nextpow2(n)
    x_padded = np.pad(x, (0, padded_size - n), 'constant')
    print(f"aaaaaaa {x_padded.shape}")

    '''
    x_padded=x.reshape(-1)-np.mean(x)
    padded_size=n
    
    # Calcola la trasformata di Fourier reale
    fft_result = np.fft.rfft(x_padded)

    # Frequenze associate alla DFT reale
    freq = np.fft.rfftfreq(padded_size,d=1/50)

    

    # Considera solo le frequenze positive
    positive_freq = freq 
    
    magnitude=np.abs(fft_result)
    #print(magnitude)
    #print(f"positive freq: {positive_freq}")
    max_mag=np.max(np.abs(magnitude[3:]))
    
    tollerance=(max_mag*5/100)
    # Trova l'indice del picco massimo
    
    indeces=np.where((magnitude <= (max_mag + tollerance)) & (magnitude > (max_mag - tollerance)))
    indeces=np.array(indeces)
    '''''
    if flag:
        print(magnitude.shape)
        plt.subplot(2, 1, 1)
        plt.plot(x_padded, marker='o')
        plt.title('Dati di input')

        plt.subplot(2, 1, 2)
        plt.plot( magnitude, 'o-')
        plt.title('Trasformata di Fourier')
        plt.xlabel('Frequenza')
        plt.ylabel('Ampiezza')
        plt.show()
    '''''
    #print(indeces)
    if len(indeces[0])>0:
        index=indeces[0][0]
    else:
        return 1,0,1
    
    reg=1
    if indeces.shape[1]>1:
        reg=regularity(indeces)
    
    #rangefreq=(np.argmax(positive_freq)-np.argmin(positive_freq))
    # Estrai la frequenza massima
    max_freq = positive_freq[index]
   
    rarity=0
    
    for i in range(3,len(positive_freq)):
        if i==index:
            continue
        D2=max_mag-magnitude[i]
        rarity+=D2
    
    ripetit=rarity*reg
    
    return ripetit,positive_freq[index],reg
def nextpow2(n):
    return int(np.ceil(np.log2(n)))
#calcolo regularity, per vedere se le ampiezze significative rilevate appaiono nella frequenza fondamentale e nei suoi multipli oppure no (rumore)
def regularity(indeces):
    indeces=indeces.T
    
    arr=[]
    arr.append(indeces[0])
    for i in range(0,indeces.shape[0]-1):
        arr.append(indeces[i+1]-indeces[i])

    arr=np.array(arr)
    max=np.max(arr)
    min=np.min(arr)
    diff=np.abs(max-min)
    
    if(diff==0):
        return 1
    else:
        return 0.1
def intersec(indici_a, indici_b):
    tolerance=50
    nuovi_indici_a = list(indici_a[0])
    for i in indici_a[0]:
        for j in range(1, tolerance + 1):
            nuovi_indici_a.extend([i - j, i + j])

    nuovi_indici_b = list(indici_b[0])
    for i in indici_b[0]:
        for j in range(1, tolerance + 1):
            nuovi_indici_b.extend([i - j, i + j])

    # Converti le liste in array numpy
    array_a = np.array(nuovi_indici_a)
    array_b = np.array(nuovi_indici_b)

    # Trova l'intersezione tra gli array numpy
    indici_and = np.intersect1d(array_a, array_b)
    indici_and=np.append(indici_and,0)
    temp=[]
    
    for i in range(0, indici_and.shape[0]):
        if i<indici_and.shape[0]-2:
            if indici_and[i+1]-indici_and[i]==1 and indici_and[i+2]-indici_and[i+1]==1 :
                temp.append(i+1)
            else:
                if temp is None:
                    continue
                else:
                    for j in temp:
                        indici_and[j]=0
                    temp.clear()
    indici=np.where(indici_and>0)
    indici=indici_and[indici] 
    return indici

def features_eng(x1,x2):
    x1=x1.reshape(-1)
    x2=x2.reshape(-1)
    v=[]
    ## 1 media  
    media1=np.mean(x1)
    media2=np.mean(x2)
    media=np.abs(media1-media2)
    v.append(media)
    
    ## 2 varianza
    var1=np.var(x1)
    var2=np.var(x2)
    var=np.abs(var1-var2)
    v.append(var)
    # 3 MAD median absolute deviation
    med1 = np.median(x1)
    mad1 = np.median(np.abs(x1 - med1))
    med2 = np.median(x2)
    mad2 = np.median(np.abs(x2 - med2))
    mad=np.abs(mad1-mad2)
    v.append(mad)
    # 4 max 
    max1=np.max(x1)
    max2=np.max(x2)
    max=np.abs(max2-max1)
    v.append(max)
    # 5 min 
    min1=np.min(x1)
    min2=np.min(x2)
    min=np.abs(min2-min1)
    v.append(min)
    # 6 sma signal magnitude area
    magnitude1 = np.sqrt(x1**2)
    sma1 = simps(magnitude1, dx=1)
    magnitude2 = np.sqrt(x2**2)
    sma2 = simps(magnitude2, dx=1)
    sma=np.abs(sma1-sma2)
    v.append(sma)
    # 7 Energy (Average sum of squares)
    squares1=np.square(x1)
    sum1=np.sum(squares1)
    mean1=sum1/x1.shape[0]
    squares2=np.square(x2)
    sum2=np.sum(squares2)
    mean2=sum2/x2.shape[0]
    En=np.abs(mean1-mean2)
    v.append(En)
    # 8 Iqr (interquartile range)
    q11=np.percentile(x1,25)
    q31=np.percentile(x1,75)
    q1=np.abs(q11-q31)
    q12=np.percentile(x2,25)
    q32=np.percentile(x2,75)
    q2=np.abs(q12-q32)
    q=np.abs(q2-q1)
    v.append(q)
    # 9 signal Entropy
    entropy1 = entropy(np.bincount(x1.astype(int)), base=2)
    entropy2 = entropy(np.bincount(x2.astype(int)), base=2)
    e=np.abs(entropy1-entropy2)
    v.append(e)
    # 10 coeff. corr (x1,x2)
    cc=np.corrcoef(x1,x2)[0,1]
    v.append(cc)
    # 11 kurtosis signal (morbidezza rispetto a gaussiana del segnale (negativo piu morbido, positivo con piu picchi e code piu veloci))
    # fisher=True => = a gaussiana =0, False => = a gaussiana =3
    fisher_kurt1 = kurtosis(x1, fisher=True)
    fisher_kurt2 = kurtosis(x2, fisher=True)
    fk=np.abs(fisher_kurt2-fisher_kurt1)
    v.append(fk)
    # 12 skewness (simmetria della distribuzione rispetto alla media)
    sk1=skew(x1)
    sk2=skew(x2)
    sk=np.abs(sk2-sk1)
    v.append(sk)
    # 13 14 kur ,skw in frequenza.
    # features on frequency 
    fft_result1 = np.fft.rfft(x1-np.mean(x1))
    #freq1 = np.fft.rfftfreq(x1.shape[0],d=1/50)   
    magnitude1=np.abs(fft_result1)
    mg1=magnitude1[3:]

    fft_result2 = np.fft.rfft(x2-np.mean(x2))
    #freq2 = np.fft.rfftfreq(x2.shape[0],d=1/50)   
    magnitude2=np.abs(fft_result2)
    mg2=magnitude2[3:]

    fisher_kurt1 = kurtosis(mg1, fisher=True)
    fisher_kurt2 = kurtosis(mg2, fisher=True)
    fk=(fisher_kurt2-fisher_kurt1)
    v.append(fk)
    
    sk1=skew(mg1)
    sk2=skew(mg2)
    sk=np.abs(sk2-sk1)
    v.append(sk)
    
    v=np.array(v)
    
    return np.array(v)

def sliding_windows1(x,k):

    ns=x.shape[0]
    
    hk=math.floor(k/2) 
    win=np.zeros((103,ns-hk))
    #np.shape(win)
   
    
    for i in range(hk,ns):
        #print(i)
        stat=[]
        #print(hk,ns)
        if i+hk>=ns:
            break
        for j in range(0,x.shape[1]): 
            if (j==1):
                continue                     
            if j>=3:
                
                w=np.vstack([x[i-hk:i,j].reshape(-1,1),x[i+1:i+hk+1,j].reshape(-1,1)])
                
                f_e=features_eng(w[:hk,:],w[hk:,:])
                rip_i, f_diff=rep_index(w[:hk,:],w[hk:,:])
                stat.append(rip_i)
                #stat.append(f_diff)
                for z in range(0,f_e.shape[0]):
                    stat.append(f_e[z])
            else:
                
                w=np.vstack([x[i-hk:i,j].reshape(-1,1),x[i+1:i+hk+1,j].reshape(-1,1)])               
                f_e=features_eng(w[:hk,:],w[hk:,:])
                for z in range(0,f_e.shape[0]):
                    stat.append(f_e[z])
                #stat.append(x[i,j])
        st=np.array(stat)
        #w=np.append(w,stat)
        #print(i)
        win[:,i]=st.reshape(-1)
        #print(win[:,i])
        del stat
        #w[:]
    return win


def samples_creation(path,path2,h):
    print(path,path2,h)
    '''
    h=75
    path="Dataset_and_Ground_Truth/data/Cora_0005.txt"
    path2="Dataset_and_Ground_Truth/Ground_Truth/Cora_0005.txt"
    '''
    ## elaborazione ground_truth
    with open(path2, 'r') as file:
            dati2 = [float(line.strip()) for line in file]
    gt = np.array(dati2).reshape(-1, 1)
    

    ground_truth=np.zeros(int(np.floor(gt[gt.shape[0]-1][0]*50))+2*h)
    '''
    print(gt.shape)
    for i in range (gt.shape[0]-1,-1,-1):
        if gt[i]==1:
            gt[i]=0
            gt[i-5]=1
            break
    k=0
    for i in range (gt.shape[0]-1,-1,-1):
        if gt[i]==1:
            k=i
            print(k)
            break
    print(k)
    ground_truth=gt[:(k+2*h)]
    '''

    for i in gt:
        index=int(np.floor(i*50))
        ground_truth[index]=1
    
    
    ## elaborazione dati
    with open(path, 'r') as file:
            dati = [line.strip().split() for line in file]
    data=np.array(dati)
    
    data=data[::4,:]
    
    data=data[:ground_truth.shape[0],:]
    
    #take=np.hstack((data,ground_truth.reshape(-1,1)))
    take=data.astype(float)
    
    samples=sliding_windows1(take,2*h).T
    samples=np.hstack((samples,ground_truth[:samples.shape[0]].reshape(-1,1)))
    print(samples.shape)
    samples=dwnsmp_and_bal(samples,h)
    print(samples.shape)
    
    return samples
def dwnsmp_and_bal(x,h):

    indexes=[]
    distances=[]

    #raccolta indici delle salienze
    for i in range(0,x.shape[0]):
        if x[i,x.shape[1]-1]==1:
            indexes.append(i)
    indexes=np.array(indexes)
    

    #raccolta distanze fra una salienza e l'altra
    for i in range(1,indexes.shape[0]):
        distances.append(indexes[i]-indexes[i-1])
    distances=np.array(distances)
    '''
    mean_v=np.mean(distances)
    means=joblib.load("saved_var/dist_m.pkl")
    
    for i in range(0,means.shape[0]):
        if means[i]==0:
            means[i]=mean_v
            break
    joblib.dump(means,"saved_var/dist_m.pkl")
    '''
    lastd=distances.shape[0]-1
    lasti=indexes.shape[0]-1 
    
    ## samples prima salienza 
    if distances[0]/2<h:
        x[indexes[0]-int(np.floor(h/2)):indexes[0]+int(np.floor(distances[0]/4)),x.shape[1]-1]=2
    else:
        x[indexes[0]-int(np.floor(h/2)):indexes[0]+int(np.floor(h/2)),x.shape[1]-1]=2
    
    pos_m=joblib.load("saved_var/pos_m.pkl")
    neg_m=joblib.load("saved_var/neg_m.pkl")

    for l in range(0,pos_m.shape[0]):
        flag=False
        for z in range(0,pos_m.shape[1]):
            if pos_m[l,z]!=0:
                flag=True
        if not flag:
            break
    ## samples dalla seconda alla penultima salienza e di tutte le "non salienze in mezzo"
    for i in range(1,lasti):
        if distances[i-1]/2<h and distances[i]/2<h:
            x[indexes[i]-int(np.floor(distances[i-1]/4)):indexes[i]+int(np.floor(distances[i]/4)),x.shape[1]-1]=1

            for z in range(0,pos_m.shape[1]):
                if pos_m[l,z]==0:
                    pos_m[l,z]=indexes[i]-int(np.floor(distances[i-1]/4))
                    break
        else:
            if distances[i-1]/2<h:
                x[indexes[i]-int(np.floor(distances[i-1]/4)):indexes[i]+int(np.floor(h/2)),x.shape[1]-1]=1

                for z in range(0,pos_m.shape[1]):
                    if pos_m[l,z]==0:
                        pos_m[l,z]=indexes[i]-int(np.floor(distances[i-1]/4))
                        break
                
            else:
                for z in range(0,pos_m.shape[1]):
                        if pos_m[l,z]==0:
                            pos_m[l,z]=indexes[i]-int(np.floor(h/2))
                            break

                if distances[i]/2<h:
                    x[indexes[i]-int(np.floor(h/2)):indexes[i]+int(np.floor(distances[i-1]/4)),x.shape[1]-1]=1

                else:
                    x[indexes[i]-int(np.floor(h/2)):indexes[i]+int(np.floor(h/2)),x.shape[1]-1]=1

                    
                    
        middle=indexes[i-1]+int(np.floor(indexes[i]-indexes[i-1])/2)
        
        if(distances[i-1]/2<h):
            x[middle-int(np.floor(distances[i-1]/4)):middle+int(np.floor(distances[i-1]/4)),x.shape[1]-1]=0
            for z in range(0,pos_m.shape[1]):
                if neg_m[l,z]==0:
                    neg_m[l,z]=middle-int(np.floor(distances[i-1]/4))
                    break
            #print(f"ciao1{x[middle-int(np.floor(distances[i-1]/4)):middle+int(np.floor(distances[i-1]/4)),x.shape[1]-1].shape}")
        else:
            x[middle-int(np.floor(h/2)):middle+int(np.floor(h/2)),x.shape[1]-1]=-1
            for z in range(0,pos_m.shape[1]):
                        if neg_m[l,z]==0:
                            neg_m[l,z]=middle-int(np.floor(h/2))
                            break
            #print(f"ciao1{x[middle-int(np.floor(h/2)):middle+int(np.floor(h/2)),x.shape[1]-1].shape}")

    ## samples dell'ultima salienza e dell'ultima non salienza (fra la penultima salienza e la penultima salienza).
    
    middle=indexes[lasti-1]+int(np.floor(indexes[lasti]-indexes[lasti-1])/2)   
    if distances[lastd]/2<h:
        x[indexes[lasti]-int(np.floor(distances[lastd]/4)):indexes[lasti]+int(np.floor(h/2)),x.shape[1]-1]=1
        x[middle-int(np.floor(distances[lastd]/4)):middle+int(np.floor(distances[lastd]/4)),x.shape[1]-1]=0
        for z in range(0,pos_m.shape[1]):
                        if neg_m[l,z]==0:
                            neg_m[l,z]=middle-int(np.floor(distances[lastd]/4))
                            break
        for z in range(0,pos_m.shape[1]):
                        if pos_m[l,z]==0:
                            pos_m[l,z]=indexes[lasti]-int(np.floor(distances[lastd]/4))
                            break
        
    else:
        x[indexes[lasti]-int(np.floor(h/2)):indexes[lasti]+int(np.floor(h/2)),x.shape[1]-1]=1
        x[middle-int(np.floor(h/2)):middle+int(np.floor(h/2)),x.shape[1]-1]=-1
        for z in range(0,pos_m.shape[1]):
                        if neg_m[l,z]==0:
                            neg_m[l,z]=middle-int(np.floor(h/2))
                            break
        for z in range(0,pos_m.shape[1]):
                        if pos_m[l,z]==0:
                            pos_m[l,z]=indexes[lasti]-int(np.floor(h/2))
                            break                    
    
    x=x[x[:,x.shape[1]-1]!=0]
    
    for i in range(0,x.shape[0]):
        if x[i,x.shape[1]-1]==-1:
            x[i,x.shape[1]-1]=0
    joblib.dump(pos_m,"saved_var/pos_m.pkl")     
    joblib.dump(pos_m,"saved_var/neg_m.pkl")   
    return x
#SPLITTING METHODS
#GENERIC SPLIT, NOT TRUSTWORTHY DUE TO THE CONTEXT!
def standard_split(samples):
    temp=[]
    for el in samples["Cora"]:
        temp.append(el)
    for el in samples["Marianne"]:
        temp.append(el)
    for el in samples["Muriel"]:
        temp.append(el)
    data=np.array(temp[0])
    print(len(temp))
    for i in range(1,len(temp)):
        data=np.vstack((data,temp[i]))
    print(data.shape)
    nl=0.7
    n_s_l = int(data.shape[0] * nl)
    indexes = np.random.permutation(data.shape[0])
    set_training = data[indexes[:n_s_l], :]
    set_test = data[indexes[n_s_l:], :]
    return set_training, set_test
#leave one sample out
def LOSO (samples): 
    TR=np.zeros((1,np.array(samples["Cora"][0]).shape[1]))
    TS=np.zeros((1,np.array(samples["Cora"][0]).shape[1]))
    for el in samples["Cora"]:        
        temp=np.array(el)
        middle=int(np.floor(temp.shape[0]/1.25))
        TR=np.vstack((TR,temp[:middle,:]))
        for i in range(middle, temp.shape[0]):
            if temp[middle,-1]==0:
                while(temp[i,-1]==0):
                    TR=np.vstack((TR,temp[i,:]))
                    i=i+1
               
                TS=np.vstack((TS,temp[i:,:]))
                break
            else:
                while(temp[i,-1]==1):
                    TR=np.vstack((TR,temp[i,:]))
                    i=i+1
                
                TS=np.vstack((TS,temp[i:,:]))
                break         
    for el in samples["Marianne"]:
        temp=np.array(el)
        middle=int(np.floor(temp.shape[0]/1.25))
        TR=np.vstack((TR,temp[:middle,:]))
        for i in range(middle, temp.shape[0]):
            if temp[middle,-1]==0:
                while(temp[i,-1]==0):
                    TR=np.vstack((TR,temp[i,:]))
                    i=i+1                             
                TS=np.vstack((TS,temp[i:,:]))
                break
            else:
                while(temp[i,-1]==1):
                    TR=np.vstack((TR,temp[i,:]))
                    i=i+1
                TS=np.vstack((TS,temp[i:,:]))
                break
    for el in samples["Muriel"]:
        temp=np.array(el)
        middle=int(np.floor(temp.shape[0]/1.25))
        TR=np.vstack((TR,temp[:middle,:]))
        for i in range(middle, temp.shape[0]):
            if temp[middle,-1]==0:
                while(temp[i,-1]==0):
                    TR=np.vstack((TR,temp[i,:]))
                    i=i+1                
                TS=np.vstack((TS,temp[i:,:]))
                break
            else:
                while(temp[i,-1]==1):
                    TR=np.vstack((TR,temp[i,:]))
                    i=i+1
                TR=np.vstack((TR,temp[i:,:]))
                break
    TS=TS[1:,:]
    print(TS.shape)
    TR=TR[1:,:]
    print(TR.shape)
    return TR, TS
#leave one take out
def LOTO (samples): 
    #1carica vettore indici
    #2random indice per ogni ballerina 
    #3confronta indici con vettore indici finche trova riga= 
    #se si hanno tris di indici accettati mettere nel test i take relativi a questi indici
    #carica nel vettore indici i 3 nuovi indici e salva il vettore.
    #return
    TR=np.zeros((1,np.array(samples["Cora"][0]).shape[1]))
    TS=np.zeros((1,np.array(samples["Cora"][0]).shape[1]))
    indexes=joblib.load("saved_var/idx_used.pkl")
    flag=False
    while flag==False:
        flag=True
        
        idx_c=np.random.randint(0,5)
        idx_ma=np.random.randint(0,12)
        idx_mu=np.random.randint(0,4)
    
        
        
        for j in range(0,indexes.shape[0]):
            if idx_c==indexes[j,0] and idx_ma==indexes[j,1] and idx_mu==indexes[j,2]:
                flag=False
                break
        
    TS=np.vstack((TS,samples["Cora"][idx_c]))
    TS=np.vstack((TS,samples["Marianne"][idx_ma]))
    TS=np.vstack((TS,samples["Muriel"][idx_mu]))
    for i in range(0,len(samples["Cora"])-1):
        if i==idx_c:
            continue
        TR=np.vstack((TR,samples["Cora"][i]))
    for i in range(0,len(samples["Marianne"])-1):
        if i==idx_ma:
            continue
        TR=np.vstack((TR,samples["Marianne"][i]))
    for i in range(0,len(samples["Muriel"])-1):
        if i==idx_mu:
            continue
        TR=np.vstack((TR,samples["Muriel"][i]))
    #temp=np.array([idx_c,idx_ma,idx_mu])
    #indexes=np.vstack((indexes,temp))
    #print(indexes)
    #joblib.dump(indexes,"saved_var/idx_used.pkl")
    TS=TS[1:,:]
    TR=TR[1:,:]
    return TR,TS
#leave one ballerina out
def LOBO (samples,j): 
    
    TR=np.zeros((1,np.array(samples["Cora"][0]).shape[1]))
    TS=np.zeros((1,np.array(samples["Cora"][0]).shape[1]))
    if j==0:
        for i in range(0,len(samples["Cora"])):
            TR=np.vstack((TR,samples["Cora"][i]))
        for i in range(0,len(samples["Marianne"])):
            TR=np.vstack((TR,samples["Marianne"][i]))
        for i in range(0,len(samples["Muriel"])):
            TS=np.vstack((TS,samples["Muriel"][i]))
    else:
        if j==1:
            for i in range(0,len(samples["Cora"])):
                TR=np.vstack((TR,samples["Cora"][i]))
            for i in range(0,len(samples["Marianne"])):
                TS=np.vstack((TS,samples["Marianne"][i]))
            for i in range(0,len(samples["Muriel"])):
                TR=np.vstack((TR,samples["Muriel"][i]))
        else:
            for i in range(0,len(samples["Cora"])):
                TS=np.vstack((TS,samples["Cora"][i]))
            for i in range(0,len(samples["Marianne"])):
                TR=np.vstack((TR,samples["Marianne"][i]))
            for i in range(0,len(samples["Muriel"])):
                TR=np.vstack((TR,samples["Muriel"][i]))
    TS=TS[1:,:]
    TR=TR[1:,:]
    return TR,TS