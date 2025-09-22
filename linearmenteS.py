# linearmenteS.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from util import plot_decision_regions

print("=" * 60)
print("EXERCÍCIO 5: DATASET LINEARMENTE SEPARÁVEL PERSONALIZADO")
print("=" * 60)

def create_custom_dataset(center_0, center_1, n_samples=50, noise=1.0, seed=42):
    """Cria dataset personalizado com dois centros"""
    np.random.seed(seed)
    # Classe 0: centro personalizado
    class_0 = np.random.randn(n_samples, 2) * noise + center_0
    # Classe 1: centro personalizado
    class_1 = np.random.randn(n_samples, 2) * noise + center_1
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    return X, y

def calculate_distance_between_centers(center_0, center_1):
    """Calcula distância euclidiana entre centros"""
    return np.sqrt(np.sum((np.array(center_1) - np.array(center_0))**2))

def analyze_decision_boundary(ppn, X_std, y):
    """Analisa a geometria da fronteira de decisão"""
    print(f"\nAnálise da Fronteira de Decisão:")
    print(f"- Peso w1 (x1): {ppn.weights[0]:.4f}")
    print(f"- Peso w2 (x2): {ppn.weights[1]:.4f}")
    print(f"- Bias: {ppn.bias:.4f}")
    
    # Calcular equação da reta: w1*x1 + w2*x2 + bias = 0
    # Rearranjando: x2 = -(w1*x1 + bias) / w2
    if abs(ppn.weights[1]) > 1e-10:  # Evitar divisão por zero
        slope = -ppn.weights[0] / ppn.weights[1]
        intercept = -ppn.bias / ppn.weights[1]
        print(f"- Equação da reta: x2 = {slope:.4f} * x1 + {intercept:.4f}")
        print(f"- Inclinação: {slope:.4f}")
        print(f"- Intercepto y: {intercept:.4f}")
    else:
        # Linha vertical
        x1_intercept = -ppn.bias / ppn.weights[0]
        print(f"- Linha vertical em x1 = {x1_intercept:.4f}")
    
    # Verificar classificação correta
    predictions = ppn.predict(X_std)
    correct_classifications = np.sum(predictions == y)
    total_points = len(y)
    print(f"- Pontos classificados corretamente: {correct_classifications}/{total_points}")
    print(f"- Todos os pontos do lado correto: {'SIM' if correct_classifications == total_points else 'NÃO'}")
    
    return slope if abs(ppn.weights[1]) > 1e-10 else None, intercept if abs(ppn.weights[1]) > 1e-10 else None

def run_experiment(center_0, center_1, title, noise=1.0):
    """Executa experimento com centros específicos"""
    print(f"\n" + "="*60)
    print(f"EXPERIMENTO: {title}")
    print(f"Centro Classe 0: {center_0}")
    print(f"Centro Classe 1: {center_1}")
    print("="*60)
    
    # Criar dataset
    X, y = create_custom_dataset(center_0, center_1, noise=noise)
    
    # Calcular distância entre centros
    distance = calculate_distance_between_centers(center_0, center_1)
    print(f"Distância entre centros: {distance:.2f}")
    print(f"Distribuição: Classe 0: {np.sum(y==0)}, Classe 1: {np.sum(y==1)}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_std = scaler.transform(X)  # Para análise completa
    
    # Treinar perceptron
    ppn = Perceptron(learning_rate=0.01, n_epochs=100)
    print(f"\nTreinando...")
    ppn.fit(X_train_std, y_train)
    
    # Avaliar
    y_pred_train = ppn.predict(X_train_std)
    y_pred_test = ppn.predict(X_test_std)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nResultados:")
    print(f"- Acurácia treino: {train_accuracy:.2%}")
    print(f"- Acurácia teste: {test_accuracy:.2%}")
    print(f"- Épocas executadas: {len(ppn.errors_history)}")
    print(f"- Convergiu: {'SIM' if 0 in ppn.errors_history else 'NÃO'}")
    if 0 in ppn.errors_history:
        conv_epoch = ppn.errors_history.index(0) + 1
        print(f"- Convergência na época: {conv_epoch}")
    
    # Análise geométrica
    slope, intercept = analyze_decision_boundary(ppn, X_std, y)
    
    return {
        'center_0': center_0,
        'center_1': center_1,
        'distance': distance,
        'X_train_std': X_train_std,
        'y_train': y_train,
        'X_std': X_std,
        'y': y,
        'ppn': ppn,
        'train_acc': train_accuracy,
        'test_acc': test_accuracy,
        'converged': 0 in ppn.errors_history,
        'slope': slope,
        'intercept': intercept
    }

# EXPERIMENTO 1: Dataset bem separado (original)
exp1 = run_experiment([-2, -2], [2, 2], "Dataset Bem Separado")

# EXPERIMENTO 2: Reduzindo a separação
exp2 = run_experiment([-1, -1], [1, 1], "Separação Reduzida")

# EXPERIMENTO 3: Separação mínima
exp3 = run_experiment([-0.5, -0.5], [0.5, 0.5], "Separação Mínima")

# EXPERIMENTO 4: Separação muito pequena (limite)
exp4 = run_experiment([-0.2, -0.2], [0.2, 0.2], "Limite de Separação")

# EXPERIMENTO 5: Diferentes orientações
exp5 = run_experiment([-2, 2], [2, -2], "Orientação Diagonal")

experiments = [exp1, exp2, exp3, exp4, exp5]

# ANÁLISE COMPARATIVA
print(f"\n" + "="*60)
print("ANÁLISE COMPARATIVA DOS EXPERIMENTOS")
print("="*60)

print(f"{'Experimento':<20} {'Distância':<10} {'Treino':<8} {'Teste':<8} {'Convergiu':<10} {'Épocas'}")
print("-" * 70)
for i, exp in enumerate(experiments, 1):
    epochs = len(exp['ppn'].errors_history)
    conv_epoch = exp['ppn'].errors_history.index(0) + 1 if exp['converged'] else epochs
    print(f"{i}. {['Bem Separado', 'Reduzida', 'Mínima', 'Limite', 'Diagonal'][i-1]:<17} "
          f"{exp['distance']:<10.2f} {exp['train_acc']:<8.1%} {exp['test_acc']:<8.1%} "
          f"{'SIM' if exp['converged'] else 'NÃO':<10} {conv_epoch}")

# VISUALIZAÇÃO COMPARATIVA
print(f"\n" + "="*60)
print("VISUALIZAÇÃO DOS EXPERIMENTOS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

titles = ['Bem Separado (d=5.66)', 'Reduzida (d=2.83)', 'Mínima (d=1.41)', 
          'Limite (d=0.57)', 'Diagonal (d=5.66)']

for i, (exp, title) in enumerate(zip(experiments, titles)):
    if i < 5:
        row = i // 3
        col = i % 3
        plt.subplot(2, 3, i+1)
        
        plot_decision_regions(exp['X_train_std'], exp['y_train'], classifier=exp['ppn'])
        
        # Adicionar linha de decisão manual para melhor visualização
        if exp['slope'] is not None:
            x_line = np.linspace(-3, 3, 100)
            y_line = exp['slope'] * x_line + exp['intercept']
            plt.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8, label='Fronteira')
        
        plt.title(f'{title}\nAcurácia: {exp["test_acc"]:.1%}')
        plt.xlabel('Feature 1 (normalizada)')
        plt.ylabel('Feature 2 (normalizada)')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

# Adicionar subplot para análise da distância
plt.subplot(2, 3, 6)
distances = [exp['distance'] for exp in experiments]
accuracies = [exp['test_acc'] for exp in experiments]
colors = ['green' if exp['converged'] else 'red' for exp in experiments]

plt.scatter(distances, accuracies, c=colors, s=100, alpha=0.7)
plt.xlabel('Distância entre Centros')
plt.ylabel('Acurácia de Teste')
plt.title('Distância vs Acurácia')
plt.grid(True, alpha=0.3)

# Adicionar labels
for i, (d, a) in enumerate(zip(distances, accuracies)):
    plt.annotate(f'{i+1}', (d, a), xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("linearmente_separavel_result.png", dpi=300, bbox_inches='tight')
#plt.show()

# ANÁLISE GEOMÉTRICA DETALHADA
print(f"\n" + "="*60)
print("ANÁLISE GEOMÉTRICA DETALHADA")
print("="*60)

print(f"\n1. GEOMETRIA DAS FRONTEIRAS DE DECISÃO:")
for i, exp in enumerate(experiments, 1):
    print(f"\nExperimento {i} - Distância: {exp['distance']:.2f}")
    if exp['slope'] is not None:
        angle = np.degrees(np.arctan(exp['slope']))
        print(f"   - Equação: x2 = {exp['slope']:.3f} * x1 + {exp['intercept']:.3f}")
        print(f"   - Ângulo com eixo x: {angle:.1f}°")
        print(f"   - Orientação: {'Positiva' if exp['slope'] > 0 else 'Negativa'}")
    
print(f"\n2. RELAÇÃO DISTÂNCIA-CONVERGÊNCIA:")
converged_experiments = [exp for exp in experiments if exp['converged']]
failed_experiments = [exp for exp in experiments if not exp['converged']]

if converged_experiments:
    min_converged_distance = min(exp['distance'] for exp in converged_experiments)
    print(f"   - Menor distância que convergiu: {min_converged_distance:.2f}")
    
if failed_experiments:
    max_failed_distance = max(exp['distance'] for exp in failed_experiments)
    print(f"   - Maior distância que NÃO convergiu: {max_failed_distance:.2f}")

print(f"\n3. ANÁLISE DA SEPARABILIDADE:")
threshold_distance = 1.0  # Baseado nos resultados
separable = [exp for exp in experiments if exp['distance'] >= threshold_distance]
challenging = [exp for exp in experiments if exp['distance'] < threshold_distance]

print(f"   - Datasets facilmente separáveis (d ≥ 1.0): {len(separable)}/5")
print(f"   - Datasets desafiadores (d < 1.0): {len(challenging)}/5")

# TESTE DO LIMITE DE SEPARABILIDADE
print(f"\n" + "="*60)
print("TESTE DO LIMITE DE SEPARABILIDADE")
print("="*60)

print(f"\nTeste com distâncias decrescentes:")
test_distances = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2]
failure_point = None

for test_dist in test_distances:
    # Calcular centros com distância específica
    center_0 = [-test_dist/2, -test_dist/2]
    center_1 = [test_dist/2, test_dist/2]
    
    X_test, y_test = create_custom_dataset(center_0, center_1, n_samples=30, noise=0.5, seed=42)
    
    # Treinar sem print para evitar poluição
    scaler = StandardScaler()
    X_test_std = scaler.fit_transform(X_test)
    
    ppn_test = Perceptron(learning_rate=0.01, n_epochs=50)
    # Temporariamente suprimir prints
    ppn_test.fit(X_test_std, y_test)
    
    accuracy = accuracy_score(y_test, ppn_test.predict(X_test_std))
    converged = 0 in ppn_test.errors_history
    
    print(f"   Distância {test_dist:.1f}: Acurácia {accuracy:.1%}, "
          f"Convergiu: {'SIM' if converged else 'NÃO'}")
    
    if not converged and failure_point is None:
        failure_point = test_dist

if failure_point:
    print(f"\n📏 PONTO DE FALHA: Distância ≈ {failure_point:.1f}")
    print(f"   Abaixo desta distância, o perceptron falha consistentemente")
else:
    print(f"\n✅ Perceptron robusto para todas as distâncias testadas")
