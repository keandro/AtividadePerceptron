# breastCancer.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from perceptron import Perceptron
from util import plot_decision_regions

print("=" * 60)
print("EXERCÍCIO 3: BREAST CANCER DATASET")
print("=" * 60)

# PASSO 1: Carregar o Dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Dataset Breast Cancer carregado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {cancer.target_names}")  # ['malignant' 'benign']
print(f"- Distribuição: Maligno: {np.sum(y==0)}, Benigno: {np.sum(y==1)}")

print(f"\nPrimeiras 10 features:")
for i in range(10):
    print(f"  {i}: {cancer.feature_names[i]}")
print(f"  ... e mais {len(cancer.feature_names)-10} features")

print(f"\n" + "="*60)
print("VERSÃO A: USANDO APENAS 2 FEATURES (VISUALIZAÇÃO)")
print("="*60)

# Versão A: Use apenas 2 features para visualização
# Vamos usar as duas primeiras features como exemplo
X_2d = X[:, [0, 1]]  # mean radius e mean texture
feature_names_2d = [cancer.feature_names[0], cancer.feature_names[1]]

print(f"Features selecionadas para visualização:")
print(f"- Feature 0: {feature_names_2d[0]}")
print(f"- Feature 1: {feature_names_2d[1]}")

# Dividir dados (Versão A)
X_train_2d, X_test_2d, y_train, y_test = train_test_split(
    X_2d, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"\nDivisão treino/teste:")
print(f"- Treino: {len(X_train_2d)} amostras")
print(f"- Teste: {len(X_test_2d)} amostras")

# Normalizar (Versão A)
scaler_2d = StandardScaler()
X_train_2d_std = scaler_2d.fit_transform(X_train_2d)
X_test_2d_std = scaler_2d.transform(X_test_2d)

# Treinar Perceptron (Versão A)
ppn_2d = Perceptron(learning_rate=0.01, n_epochs=100)
print(f"\nTreinando com 2 features...")
ppn_2d.fit(X_train_2d_std, y_train)

# Avaliar (Versão A)
y_pred_train_2d = ppn_2d.predict(X_train_2d_std)
y_pred_test_2d = ppn_2d.predict(X_test_2d_std)

train_accuracy_2d = accuracy_score(y_train, y_pred_train_2d)
test_accuracy_2d = accuracy_score(y_test, y_pred_test_2d)

print(f"\nResultados - VERSÃO A (2 features):")
print(f"- Acurácia no treino: {train_accuracy_2d:.2%}")
print(f"- Acurácia no teste: {test_accuracy_2d:.2%}")
print(f"- Convergiu: {'SIM' if 0 in ppn_2d.errors_history else 'NÃO'}")
print(f"- Épocas executadas: {len(ppn_2d.errors_history)}")

print(f"\n" + "="*60)
print("VERSÃO B: USANDO TODAS AS 30 FEATURES")
print("="*60)

# Versão B: Usar todas as 30 features
X_full = X  # Todas as features

# Dividir dados (Versão B)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Usando todas as {X_full.shape[1]} features:")
print(f"- Divisão: {len(X_train_full)} treino, {len(X_test_full)} teste")

# Normalizar (Versão B)
scaler_full = StandardScaler()
X_train_full_std = scaler_full.fit_transform(X_train_full)
X_test_full_std = scaler_full.transform(X_test_full)

print(f"- Normalização aplicada (importante para 30 features!)")

# Treinar Perceptron (Versão B)
ppn_full = Perceptron(learning_rate=0.01, n_epochs=100)
print(f"\nTreinando com todas as 30 features...")
ppn_full.fit(X_train_full_std, y_train_full)

# Avaliar (Versão B)
y_pred_train_full = ppn_full.predict(X_train_full_std)
y_pred_test_full = ppn_full.predict(X_test_full_std)

train_accuracy_full = accuracy_score(y_train_full, y_pred_train_full)
test_accuracy_full = accuracy_score(y_test_full, y_pred_test_full)

print(f"\nResultados - VERSÃO B (30 features):")
print(f"- Acurácia no treino: {train_accuracy_full:.2%}")
print(f"- Acurácia no teste: {test_accuracy_full:.2%}")
print(f"- Convergiu: {'SIM' if 0 in ppn_full.errors_history else 'NÃO'}")
print(f"- Épocas executadas: {len(ppn_full.errors_history)}")

print(f"\n" + "="*60)
print("COMPARAÇÃO ENTRE VERSÕES:")
print("="*60)

print(f"VERSÃO A (2 features):")
print(f"  - Acurácia teste: {test_accuracy_2d:.2%}")
print(f"  - Convergiu: {'SIM' if 0 in ppn_2d.errors_history else 'NÃO'}")
print(f"  - Epochs: {len(ppn_2d.errors_history)}")

print(f"\nVERSÃO B (30 features):")
print(f"  - Acurácia teste: {test_accuracy_full:.2%}")
print(f"  - Convergiu: {'SIM' if 0 in ppn_full.errors_history else 'NÃO'}")
print(f"  - Epochs: {len(ppn_full.errors_history)}")

print(f"\nMELHORIA:")
improvement = test_accuracy_full - test_accuracy_2d
print(f"  - Diferença: {improvement:.2%}")
print(f"  - Mais features = {'MELHOR' if improvement > 0 else 'PIOR'} resultado")

print(f"\n" + "="*60)
print("ANÁLISE DETALHADA - MÉTRICAS MÉDICAS:")
print("="*60)

print(f"\nClassification Report - VERSÃO A (2 features):")
print(f"-" * 50)
print(classification_report(y_test, y_pred_test_2d, 
                          target_names=cancer.target_names,
                          zero_division=0))

print(f"\nClassification Report - VERSÃO B (30 features):")
print(f"-" * 50)
print(classification_report(y_test_full, y_pred_test_full, 
                          target_names=cancer.target_names,
                          zero_division=0))

print(f"\n" + "="*60)
print("MATRIZ DE CONFUSÃO:")
print("="*60)

# Matriz de confusão para Versão B (melhor resultado)
cm = confusion_matrix(y_test_full, y_pred_test_full)
print(f"\nMatriz de Confusão - VERSÃO B (30 features):")
print(f"                    Predito")
print(f"                Maligno  Benigno")
print(f"Real  Maligno  {cm[0,0]:8d}  {cm[0,1]:7d}")
print(f"      Benigno  {cm[1,0]:8d}  {cm[1,1]:7d}")

# Análise médica
tn, fp, fn, tp = cm.ravel()
print(f"\nAnálise Médica:")
print(f"- Verdadeiros Negativos (TN): {tn} (benigno predito como benigno)")
print(f"- Falsos Positivos (FP): {fp} (benigno predito como maligno)")
print(f"- Falsos Negativos (FN): {fn} (maligno predito como benigno) ⚠️ CRÍTICO!")
print(f"- Verdadeiros Positivos (TP): {tp} (maligno predito como maligno)")

# Métricas médicas importantes
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall para malignos
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall para benignos

print(f"\nMétricas Médicas Críticas:")
print(f"- Sensibilidade (detectar malignos): {sensitivity:.2%}")
print(f"- Especificidade (detectar benignos): {specificity:.2%}")
print(f"- Taxa de Falsos Negativos: {fn/(fn+tp):.2%} ⚠️")
print(f"- Taxa de Falsos Positivos: {fp/(fp+tn):.2%}")

print(f"\nIMPORTÂNCIA MÉDICA:")
if fn > 0:
    print(f"⚠️  ATENÇÃO: {fn} casos malignos foram classificados como benignos!")
    print(f"   Isso pode ser PERIGOSO em diagnóstico médico!")
else:
    print(f"✅ Excelente: Nenhum caso maligno foi perdido!")

# Visualização para Versão A (2 features)
print(f"\n" + "="*60)
print("VISUALIZAÇÃO (VERSÃO A - 2 FEATURES):")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Regiões de Decisão
plt.subplot(2, 2, 1)
plot_decision_regions(X_train_2d_std, y_train, classifier=ppn_2d)
plt.title('Regiões de Decisão - Breast Cancer (2D)')
plt.xlabel(f'{feature_names_2d[0]} (normalizado)')
plt.ylabel(f'{feature_names_2d[1]} (normalizado)')
plt.legend(loc='upper right')

# Subplot 2: Convergência Versão A
plt.subplot(2, 2, 2)
plt.plot(range(1, len(ppn_2d.errors_history) + 1), ppn_2d.errors_history, 
         marker='o', color='red', label='2 features')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência - 2 Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3: Convergência Versão B
plt.subplot(2, 2, 3)
plt.plot(range(1, len(ppn_full.errors_history) + 1), ppn_full.errors_history, 
         marker='s', color='blue', label='30 features')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência - 30 Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 4: Matriz de Confusão
plt.subplot(2, 2, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title('Matriz de Confusão (30 features)')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')

plt.tight_layout()
plt.savefig("breast_cancer_result.png", dpi=300, bbox_inches='tight')
#plt.show()

print(f"\n" + "="*60)
print("CONCLUSÕES FINAIS:")
print("="*60)

print(f"1. COMPARAÇÃO DE PERFORMANCE:")
print(f"   - 2 features: {test_accuracy_2d:.2%} acurácia")
print(f"   - 30 features: {test_accuracy_full:.2%} acurácia")
print(f"   - Melhoria: {improvement:.2%}")

print(f"\n2. ADEQUAÇÃO MÉDICA:")
if test_accuracy_full >= 0.95:
    print(f"   ✅ Acurácia excelente para aplicação médica")
else:
    print(f"   ⚠️ Acurácia pode não ser suficiente para diagnóstico")

print(f"\n3. FALSOS NEGATIVOS (CRÍTICO):")
if fn == 0:
    print(f"   ✅ Nenhum caso maligno perdido - IDEAL!")
elif fn <= 2:
    print(f"   ⚠️ Poucos casos malignos perdidos - ACEITÁVEL")
else:
    print(f"   ❌ Muitos casos malignos perdidos - PERIGOSO!")

print(f"\n4. RECOMENDAÇÕES:")
print(f"   - Use TODAS as features disponíveis")
print(f"   - Considere algoritmos mais robustos (SVM, Random Forest)")
print(f"   - Sempre avalie falsos negativos em problemas médicos")
print(f"   - Implemente validação cruzada para maior confiabilidade")

print(f"\n5. LIMITAÇÕES DO PERCEPTRON:")
print(f"   - Assume separabilidade linear")
print(f"   - Pode não capturar relações complexas entre features")
print(f"   - Para diagnóstico médico, considere métodos ensemble")