# ruido.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from util import plot_decision_regions

print("=" * 60)
print("EXERCÍCIO 4: DATASET DE CLASSIFICAÇÃO COM RUÍDO")
print("=" * 60)

# Classe Perceptron com Early Stopping
class PerceptronEarlyStopping(Perceptron):
    def __init__(self, learning_rate=0.01, n_epochs=100, patience=10):
        super().__init__(learning_rate, n_epochs)
        self.patience = patience
        self.validation_history = []
        
    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        # PASSO 1: Inicialização dos pesos
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Variáveis para early stopping
        best_val_accuracy = 0
        patience_counter = 0
        best_weights = None
        best_bias = None

        # PASSO 2: Loop de treinamento
        for epoch in range(self.n_epochs):
            errors = 0

            # PASSO 3: Para cada exemplo de treinamento
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                error = y[idx] - y_predicted
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0)
            
            self.errors_history.append(errors)
            
            # Early Stopping com validação
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                self.validation_history.append(val_accuracy)
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping na época {epoch + 1} (paciência: {self.patience})")
                    # Restaurar melhores pesos
                    self.weights = best_weights
                    self.bias = best_bias
                    break
            
            # Parada antecipada se convergiu
            if errors == 0:
                print(f"Convergiu na época {epoch + 1}")
                break

def run_experiment(class_sep, flip_y, title):
    print(f"\n" + "="*60)
    print(f"EXPERIMENTO: {title}")
    print(f"class_sep={class_sep}, flip_y={flip_y}")
    print("="*60)
    
    # Gerar dataset
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,  # Controla separação
        flip_y=flip_y,        # % de ruído nos rótulos
        random_state=42
    )
    
    print(f"Dataset gerado:")
    print(f"- Amostras: {X.shape[0]}")
    print(f"- Features: {X.shape[1]}")
    print(f"- Separação entre classes: {class_sep}")
    print(f"- Ruído nos rótulos: {flip_y*100:.1f}%")
    print(f"- Distribuição: Classe 0: {np.sum(y==0)}, Classe 1: {np.sum(y==1)}")
    
    # Dividir em treino/validação/teste (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )
    
    print(f"Divisão: {len(X_train)} treino, {len(X_val)} validação, {len(X_test)} teste")
    
    # Normalizar
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    # Treinar com Early Stopping
    ppn = PerceptronEarlyStopping(learning_rate=0.01, n_epochs=100, patience=10)
    ppn.fit(X_train_std, y_train, X_val_std, y_val)
    
    # Avaliar
    y_pred_train = ppn.predict(X_train_std)
    y_pred_val = ppn.predict(X_val_std)
    y_pred_test = ppn.predict(X_test_std)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nResultados:")
    print(f"- Acurácia treino: {train_accuracy:.2%}")
    print(f"- Acurácia validação: {val_accuracy:.2%}")
    print(f"- Acurácia teste: {test_accuracy:.2%}")
    print(f"- Épocas executadas: {len(ppn.errors_history)}")
    print(f"- Convergiu: {'SIM' if 0 in ppn.errors_history else 'NÃO'}")
    
    # Análise do ruído
    noise_impact = len(X) * flip_y
    print(f"\nImpacto do ruído:")
    print(f"- Amostras com rótulos incorretos: ~{noise_impact:.0f}")
    print(f"- Impacto na acurácia máxima teórica: ~{(1-flip_y)*100:.1f}%")
    
    return {
        'class_sep': class_sep,
        'flip_y': flip_y,
        'train_acc': train_accuracy,
        'val_acc': val_accuracy,
        'test_acc': test_accuracy,
        'epochs': len(ppn.errors_history),
        'converged': 0 in ppn.errors_history,
        'X_train_std': X_train_std,
        'y_train': y_train,
        'ppn': ppn
    }

# EXPERIMENTO 1: Variando separação entre classes
print(f"\n" + "🔬"*30)
print("PARTE 1: VARIANDO SEPARAÇÃO ENTRE CLASSES")
print("🔬"*30)

experiments_sep = []
class_seps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
flip_y_fixed = 0.05

for class_sep in class_seps:
    result = run_experiment(class_sep, flip_y_fixed, f"Separação {class_sep}")
    experiments_sep.append(result)

# EXPERIMENTO 2: Variando ruído nos rótulos
print(f"\n" + "🎯"*30)
print("PARTE 2: VARIANDO RUÍDO NOS RÓTULOS")
print("🎯"*30)

experiments_noise = []
flip_ys = [0.0, 0.05, 0.10, 0.15, 0.20]
class_sep_fixed = 1.5

for flip_y in flip_ys:
    result = run_experiment(class_sep_fixed, flip_y, f"Ruído {flip_y*100:.0f}%")
    experiments_noise.append(result)

# ANÁLISE COMPARATIVA
print(f"\n" + "="*60)
print("ANÁLISE COMPARATIVA DOS EXPERIMENTOS")
print("="*60)

print(f"\n1. EFEITO DA SEPARAÇÃO ENTRE CLASSES (ruído fixo = {flip_y_fixed*100:.0f}%):")
print(f"{'Separação':<10} {'Treino':<8} {'Validação':<10} {'Teste':<8} {'Épocas':<8} {'Convergiu'}")
print("-" * 60)
for exp in experiments_sep:
    print(f"{exp['class_sep']:<10.1f} {exp['train_acc']:<8.1%} "
          f"{exp['val_acc']:<10.1%} {exp['test_acc']:<8.1%} "
          f"{exp['epochs']:<8d} {'SIM' if exp['converged'] else 'NÃO'}")

print(f"\n2. EFEITO DO RUÍDO NOS RÓTULOS (separação fixa = {class_sep_fixed}):")
print(f"{'Ruído':<10} {'Treino':<8} {'Validação':<10} {'Teste':<8} {'Épocas':<8} {'Convergiu'}")
print("-" * 60)
for exp in experiments_noise:
    print(f"{exp['flip_y']*100:<10.0f}% {exp['train_acc']:<8.1%} "
          f"{exp['val_acc']:<10.1%} {exp['test_acc']:<8.1%} "
          f"{exp['epochs']:<8d} {'SIM' if exp['converged'] else 'NÃO'}")

# VISUALIZAÇÃO
print(f"\n" + "="*60)
print("VISUALIZAÇÃO DOS EXPERIMENTOS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Primeira linha: Variação de separação
for i, exp in enumerate(experiments_sep[::2]):  # Mostrar apenas alguns para clareza
    plt.subplot(2, 3, i+1)
    plot_decision_regions(exp['X_train_std'], exp['y_train'], classifier=exp['ppn'])
    plt.title(f'Separação: {exp["class_sep"]}\nAcurácia: {exp["test_acc"]:.1%}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# Segunda linha: Variação de ruído
for i, exp in enumerate(experiments_noise[::2]):  # Mostrar apenas alguns para clareza
    if i < 3:
        plt.subplot(2, 3, i+4)
        plot_decision_regions(exp['X_train_std'], exp['y_train'], classifier=exp['ppn'])
        plt.title(f'Ruído: {exp["flip_y"]*100:.0f}%\nAcurácia: {exp["test_acc"]:.1%}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

plt.tight_layout()
plt.savefig("ruido_result.png", dpi=300, bbox_inches='tight')
#plt.show()

# GRÁFICOS DE ANÁLISE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: Separação vs Acurácia
plt.subplot(1, 2, 1)
seps = [exp['class_sep'] for exp in experiments_sep]
test_accs_sep = [exp['test_acc'] for exp in experiments_sep]
plt.plot(seps, test_accs_sep, marker='o', linewidth=2, markersize=8)
plt.xlabel('Separação entre Classes')
plt.ylabel('Acurácia de Teste')
plt.title('Impacto da Separação na Performance')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

# Gráfico 2: Ruído vs Acurácia
plt.subplot(1, 2, 2)
noises = [exp['flip_y']*100 for exp in experiments_noise]
test_accs_noise = [exp['test_acc'] for exp in experiments_noise]
plt.plot(noises, test_accs_noise, marker='s', linewidth=2, markersize=8, color='red')
plt.xlabel('Ruído nos Rótulos (%)')
plt.ylabel('Acurácia de Teste')
plt.title('Impacto do Ruído na Performance')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig("ruido_analysis.png", dpi=300, bbox_inches='tight')
#plt.show()

# CONCLUSÕES FINAIS
print(f"\n" + "="*60)
print("CONCLUSÕES FINAIS:")
print("="*60)

print(f"\n1. EFEITO DA SEPARAÇÃO ENTRE CLASSES:")
best_sep = max(experiments_sep, key=lambda x: x['test_acc'])
worst_sep = min(experiments_sep, key=lambda x: x['test_acc'])
print(f"   - Melhor separação: {best_sep['class_sep']} (acurácia: {best_sep['test_acc']:.1%})")
print(f"   - Pior separação: {worst_sep['class_sep']} (acurácia: {worst_sep['test_acc']:.1%})")
print(f"   - Diferença: {best_sep['test_acc'] - worst_sep['test_acc']:.1%}")

print(f"\n2. EFEITO DO RUÍDO NOS RÓTULOS:")
best_noise = max(experiments_noise, key=lambda x: x['test_acc'])
worst_noise = min(experiments_noise, key=lambda x: x['test_acc'])
print(f"   - Melhor ruído: {best_noise['flip_y']*100:.0f}% (acurácia: {best_noise['test_acc']:.1%})")
print(f"   - Pior ruído: {worst_noise['flip_y']*100:.0f}% (acurácia: {worst_noise['test_acc']:.1%})")
print(f"   - Diferença: {best_noise['test_acc'] - worst_noise['test_acc']:.1%}")

print(f"\n3. EARLY STOPPING:")
early_stopped = sum(1 for exp in experiments_sep + experiments_noise if not exp['converged'])
total_exp = len(experiments_sep) + len(experiments_noise)
print(f"   - Experimentos que usaram early stopping: {early_stopped}/{total_exp}")
print(f"   - Útil para evitar overfitting em dados ruidosos")