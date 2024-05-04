from library import *
'''
LOSO=joblib.load("saved_var/test_evalLOSO.pkl")
LOTO=joblib.load("saved_var/test_evalLOTO.pkl")
LOBO=joblib.load("saved_var/test_evalLOBO.pkl")
LOSO_pt=joblib.load("saved_var/test_evalLOSO_PT.pkl")
LOTO_pt=joblib.load("saved_var/test_evalLOTO_PT.pkl")
LOBO_pt=joblib.load("saved_var/test_evalLOBO_PT.pkl")
print("accuracy precision recall f1_score")
print(f"LOSO {LOSO,LOSO_pt}")
print(f"LOTO {LOTO,LOTO_pt}")
print(f"LOBO {LOBO,LOBO_pt}")
'''

LOBO=joblib.load("saved_var/best_rf_val_LOBO.pkl")
LOBO=LOBO[:-1]
mean_LOBO=np.zeros((1,4))
var_LOBO=np.zeros((1,4))
for i in range(0,LOBO.shape[1]):
    mean_LOBO[0,i]=np.mean(LOBO[:,i])
    var_LOBO[0,i]=np.var(LOBO[:,i])
print(f"LOBO {LOBO}")
print(f"mean {mean_LOBO}")
print(f"variance {var_LOBO}")

LOTO=joblib.load("saved_var/best_rf_val_LOTO.pkl")
LOTO=LOTO[:-1]
mean_LOTO=np.zeros((1,4))
var_LOTO=np.zeros((1,4))
for i in range(0,LOTO.shape[1]):
    mean_LOTO[0,i]=np.mean(LOTO[:,i])
    var_LOTO[0,i]=np.var(LOTO[:,i])
print(f"LOTO {LOTO}")
print(f"mean {mean_LOTO}")
print(f"variance {var_LOTO}")

pos_m=joblib.load("saved_var/pos_m.pkl")
neg_m=joblib.load("saved_var/neg_m.pkl")
print(f"p {pos_m}")
print(f"n {neg_m}")