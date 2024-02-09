from library import *
from functions import *
from dataset import *
k=151
h=int(np.floor(k/2))
means=np.zeros((21,1))
pos_m=np.zeros((21,80))
neg_m=np.zeros((21,80))
joblib.dump(pos_m,"saved_var/pos_m.pkl")
joblib.dump(pos_m,"saved_var/neg_m.pkl")
samples=import_samples(h)

print(means)

#print(np.mean(means))
joblib.dump(samples,"saved_var/samples_3.pkl")
#print(dataset["Cora"][0])