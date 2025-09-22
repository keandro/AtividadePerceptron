# iris.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from util import plot_decision_regions

# PASSO 1: Carregar o Dataset Iris
print("=" * 50)
print("EXERCÍCIO 1: IRIS DATASET (Setosa vs Versicolor)")
print("=" * 50)

iris = datasets.load_iris()
# IMPORTANTE: Use apenas as classes 0 e 1 (Setosa e Versicolor)
# Classe 2 (Virginica) não é linearmente separável das outras
mask = iris.target != 2
X = iris.data[mask]
y = iris.target[mask]
# Sugestão: Use apenas 2 features para visualização
# Por exemplo: índices [0, 2] = comprimento da sépala e comprimento da pétala
X = X[:, [0, 2]]

print(f"Dataset Iris carregado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]} (comprimento sépala e pétala)")
print(f"- Classes: {np.unique(y)} (0=Setosa, 1=Versicolor)")
print(f"- Distribuição das classes: Classe 0: {np.sum(y==0)}, Classe 1: {np.sum(y==1)}")

# PASSO 2: Dividir em Treino e Teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,  # 30% para teste
    random_state=42,
    stratify=y  # Mantém proporção das classes
)

print(f"\nDivisão treino/teste:")
print(f"- Treino: {len(X_train)} amostras")
print(f"- Teste: {len(X_test)} amostras")

# PASSO 3: Normalização
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

print(f"\nDados normalizados (z-score)")

# PASSO 4: Treinar o Perceptron
ppn = Perceptron(learning_rate=0.01, n_epochs=50)
print(f"\nIniciando treinamento...")
ppn.fit(X_train_std, y_train)

# PASSO 5: Avaliar o Modelo
y_pred_train = ppn.predict(X_train_std)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = ppn.predict(X_test_std)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Acurácia total
y_pred_total = np.concatenate([y_pred_train, y_pred_test])
y_total = np.concatenate([y_train, y_test])
total_accuracy = accuracy_score(y_total, y_pred_total)

# Erros
train_errors = np.sum(y_pred_train != y_train)
test_errors = np.sum(y_pred_test != y_test)
total_errors = int(train_errors + test_errors)

print(f"\nResultados:")
print(f"- Acurácia no conjunto de treinamento: {train_accuracy:.2%}")
print(f"- Acurácia no conjunto de teste: {test_accuracy:.2%}")
print(f"- Acurácia em todo o conjunto: {total_accuracy:.2%}")
print(f"- Total de amostras classificadas erradas: {total_errors}")
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")

# Verificar convergência
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0) + 1
    print(f"- Convergiu na época: {conv_epoch}")
else:
    print("- Não convergiu completamente")
    
print(f"- Total de épocas executadas: {len(ppn.errors_history)}")

# PASSO 6: Análise dos Pesos Aprendidos
print(f"\nPesos aprendidos:")
print(f"- w1 (comprimento sépala): {ppn.weights[0]:.4f}")
print(f"- w2 (comprimento pétala): {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

# Equação da fronteira de decisão
if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")

# PASSO 7: Visualizar Resultados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Regiões de Decisão
plt.subplot(1, 2, 1)
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Regiões de Decisão - Iris (Setosa vs Versicolor)')
plt.xlabel('Comprimento Sépala (normalizado)')
plt.ylabel('Comprimento Pétala (normalizado)')
plt.legend(loc='upper left')

# Subplot 2: Curva de Convergência
plt.subplot(1, 2, 2)
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento - Iris')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("iris_result.png")
#plt.show()
