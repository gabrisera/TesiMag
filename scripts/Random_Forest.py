from library import *
from functions import *
from dataset import *
samples=joblib.load("saved_var/samples_2.pkl")
'''
indexes=np.zeros((1,3))
for j in range(0,indexes.shape[1]):
    indexes[0,j]=100
'''
values=joblib.load("saved_var/best_rf_val_LOTO.pkl")
#values=np.zeros((1,4))
#joblib.dump(indexes,"saved_var/idx_used.pkl")
###
#for i in range(0,10):
trs,ts=LOTO(samples)
# Supponendo che l'ultima colonna sia la variabile da predire
X_train = trs[:, :-1]
y_train = trs[:, -1]
print_val_count(y_train)
X_test = ts[:, :-1]
y_test = ts[:, -1]
print_val_count(y_test)
# Creare un classificatore Random Forest
random_forest = RandomForestClassifier(random_state=42)

parametri_grid = {
    'n_estimators': [1000],
    'min_samples_leaf': [1,2],
    'max_features': [0.1,0.2],
    'bootstrap': [True]
}
scorer = make_scorer(f1_score)
grid_search = GridSearchCV(random_forest, parametri_grid, cv=3, scoring=scorer, verbose=2)

# Addestrare il modello sulla matrice di training
grid_search.fit(X_train, y_train)
#best_params=grid_search.best_params_
#random_forest.fit(X_train,y_train)
# Effettuare previsioni sulla matrice di test
predictions = grid_search.best_estimator_.predict(X_test)
#predictions= random_forest.predict(X_test)
# Calcolare l'accuratezza del modello
accuracy = accuracy_score(y_test, predictions)
precision=precision_score(y_test, predictions)
recall=recall_score(y_test, predictions)
f1=f1_score(y_test, predictions)
eval=[accuracy, precision, recall, f1]
print(eval)
eval=np.array(eval)
values=np.vstack((values,eval))
print(values)

#values=values[1:,:]
joblib.dump(values,"saved_var/best_rf_val_LOTO.pkl")

#best_svr = grid_search.best_estimator_
#best_params = grid_search.best_params_
#joblib.dump(best_svr,"saved_var/best_svr_model.pkl")
#joblib.dump(best_params,"saved_var/best_svr_params_LOBO_120prova.pkl")
joblib.dump(predictions,"saved_var/test_predictionsLOBO200prova.pkl")
joblib.dump(eval,"saved_var/test_evalLOBO200prova.pkl")
print(f'accuracy, precision, recall, f1: {accuracy,precision,recall,f1}')