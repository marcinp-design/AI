# 1. Wczytanie bibliotek matplotlib, pandas, numpy, seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 2. Wczytanie pliku CSV
df=pd.read_csv('C:/Users/marci/Downloads/pulsar_stars.csv')
print(df.shape)
# 3. Zmienić nazwy kolumn - usunać znaki białe (spacje) z początku nazwk kolumn 
df.columns = df.columns.str.strip()
print("Kolumny:", list(df.columns))
print(df.head(), "\n")
# 4. Atrybut, który przewidujemy to kolumna target_class - zapoznać się z jej dystrybucją (ile wartości jest 0, ile 1) 
print("Rozkład target_class:")
print(df["target_class"].value_counts())
print(df["target_class"].value_counts(normalize=True).round(4), "\n")
 
sns.countplot(x="target_class", data=df)
plt.title("Rozkład klas (target_class)")
plt.show()
# 5. Sprawdzić czy istnieją wartości puste. 
print(10*'=' + 'Sprawdzić czy istnieją wartości puste' + 10*'=')
print(df.isna().sum())
df = df.dropna()

# 6. Policzyć statystyki opisowe dla kolumn - użyć funkcji describe obiektu DataFrame
print(df.describe().T, "\n")

# 7. Dokonać wizualizacji wartości ostających (outlier'ów) - użyć boxplot
plt.figure(figsize=(15, 5))
sns.boxplot(data=df.drop(columns="target_class"), orient="h")
plt.title("Boxplot cech (wartości odstające)")
plt.tight_layout()
plt.show()

# 8. Wygenerować histogramy dla każdej z kolumn 
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histogramy dla każdej z kolumn")
plt.tight_layout()
plt.show()

# 9. Obiekt df podzielić na cechy i atrybut, który przewidujemy (X i y) 
X = df.drop(columns="target_class")
y = df["target_class"]

# 10. Podzielić dane na osobny zbiór treningowy i testowy 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30, stratify=y
)
 # test_size=0.2 – 20% danych trafia do zbioru testowego, a 80% do treningowego
 # random_state=30 – ustala promien losowania
 # stratify=y – zachowuje proporcje klas w obu zbiorach.

 # 11. Dokonać skalowania cech (użyć StandardScaler) - na zbiorze treningowym użyć fit_transform, na zbiorze testowym użyc transform 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 12. Dopasować model SVC (parametry domyślne): svc.fit(X_train,y_train) oraz y_pred=svc.predict(X_test).
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# 13. Policz efektywność predykcji (accuracy_score)
from sklearn.metrics import accuracy_score 
print(f"SVC, accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 14. Utwórz nowy model SVC, ze zwiększonym parametrem C równym 100
svc_c100 = SVC(C=100)
svc_c100.fit(X_train, y_train)
y_pred_c100 = svc_c100.predict(X_test)
#ponownie policz efektywność predykcji (accuracy_score)  
print(f"SVC (C=100), accuracy: {accuracy_score(y_test, y_pred_c100):.4f}") 

# 15. Porównaj jakosć predykcji na zbiorze treningowym i testowym - linear_svc. Użyj funkcji score. 
linear_svc = SVC(kernel="linear")
linear_svc.fit(X_train, y_train)
print(f"linear_svc: train score = {linear_svc.score(X_train, y_train):.4f}, "
      f"test score = {linear_svc.score(X_test, y_test):.4f}")

#16. Sprawdź inne warianty parametrów SVC - powtórz kroki powyżej z kernel='poly', kernel='sigmoid'

for kernel in [ "poly", "sigmoid"]:
    for C in [1, 100]:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        y_p = model.predict(X_test)
        print(f"kernel={kernel:8s} C={C:<4} "
              f"train={model.score(X_train, y_train):.4f}  "
              f"test={accuracy_score(y_test, y_p):.4f}")

# 17. Wyświetl macierz pomyłek - confusion_matrix 
from sklearn.metrics import  confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred_c100)
print("\nMacierz pomyłek:\n", cm)
print("\n", classification_report(y_test, y_pred_c100))
# 18. Dokonaj wizualizacji macierz pomyłek z użyciem heatmap 
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz pomyłek - SVC (C=100)")
plt.tight_layout()
plt.show()
# 19. Dokonaj optymalizacji hiperparametrów z użyciem GridSearchCV. 
from sklearn.model_selection import GridSearchCV
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", 0.01, 0.1],
}

#"C": [0.1, 1, 10, 100] – siła kary za błędy klasyfikacji. Małe C daje prostszy model, który toleruje więcej pomyłek. Duże C mocniej dopasowuje się do danych treningowych, co może prowadzić do przeuczenia.
# "kernel": ["linear", "rbf"] – kształt granicy między klasami. linear rysuje prostą granicę (płaszczyznę), a rbf potrafi tworzyć granice zakrzywione, bardziej złożone.
# "gamma": ["scale", 0.01, 0.1] – zasięg wpływu pojedynczego punktu treningowego.

grid = GridSearchCV(SVC(), param_grid, cv=2, scoring="accuracy", n_jobs=-1, verbose=1)
#cv=3 – 2-krotna walidacja krzyżowa. Dane treningowe są dzielone na 2 części. Model trenuje się na 1 częściach i sprawdza na drugiej.
#scoring="accuracy" – miara, według której wybierany jest najlepszy model, tutaj odsetek poprawnych klasyfikacji.
#n_jobs=-1 – użyj wszystkich rdzeni procesora równolegle, co przyspiesza obliczenia
#verbose=1 – wypisuje krótką informację o postępie,
grid.fit(X_train, y_train)
 
print("\nNajlepsze parametry:", grid.best_params_)
print(f"Najlepsza accuracy (walidacja krzyżowa): {grid.best_score_:.4f}")
 
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Accuracy najlepszego modelu na zbiorze testowym: {accuracy_score(y_test, y_pred_best):.4f}")
 
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz pomyłek - najlepszy model (GridSearchCV)")
plt.tight_layout()
plt.show()

# inne obliczenia
grid2 = GridSearchCV(SVC(), param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid2.fit(X_train, y_train)
 
print("\nNajlepsze parametry:", grid2.best_params_)
print(f"Najlepsza accuracy (walidacja krzyżowa): {grid2.best_score_:.4f}")
 
best_model_2 = grid2.best_estimator_
y_pred_best_2 = best_model_2.predict(X_test)
print(f"Accuracy najlepszego modelu na zbiorze testowym: {accuracy_score(y_test, y_pred_best_2):.4f}")
 
cm_best = confusion_matrix(y_test, y_pred_best_2)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz pomyłek - najlepszy model (GridSearchCV)")
plt.tight_layout()
plt.show()