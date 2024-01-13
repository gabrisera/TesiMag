from library import *
from functions import *
cartella = 'Ground_Truth_and_videos'

# Lista per contenere i vettori colonna NumPy da ciascun file
vettori_colonna = []

# Iterare attraverso i file nella cartella
for nome_file in os.listdir(cartella):
    if nome_file.endswith('.txt'):  # Assicurarsi che il file sia un file di testo
        percorso_file = os.path.join(cartella, nome_file)

        # Leggere i dati dal file e creare un vettore colonna NumPy
        with open(percorso_file, 'r') as file:
            dati = [float(line.strip()) for line in file]

        vettore_colonna = np.array(dati).reshape(-1, 1)
        print(vettore_colonna.shape)
        ground_truth=np.zeros(int(np.floor(vettore_colonna[vettore_colonna.shape[0]-1][0]*50)))
        print(f"ground_truth.dim {ground_truth.shape}")
        for i in vettore_colonna:
            if i==vettore_colonna[vettore_colonna.shape[0]-1][0]:
                continue
            index=int(np.floor(i*50))
            ground_truth[index]=1
        # Elaborare i dati qui (puoi aggiungere la tua logica di elaborazione)
        percorso_file="Ground_Truth_and_videos/frames/"+"frame_"+nome_file
        np.savetxt(percorso_file, ground_truth.reshape(-1, 1), fmt='%d')
        print_val_count(ground_truth)
        # Stampare o fare qualcos'altro con il vettore colonna elaborato
        print(f"Dati elaborati da {nome_file}:")
        print(vettore_colonna)

        # Svuotare l'array per il prossimo file
        vettori_colonna = []