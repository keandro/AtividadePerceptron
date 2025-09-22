# Atividade Prática de Inteligência Artificial

## Dupla

- Leandro Carvalho
- Yuri Carneiro

## 1. Iris

1. Descrição do DatasetDescrição do Dataset:
    - Número de amostras: 100 (50 Setosa + 50 Versicolor)
    - Número de features: 2 (comprimento sépala e pétala)
    - Distribuição das classes: balanceada (50/50)
    - É linearmente separável? SIM (classes 0 e 1 são separáveis)

2. Resultados Obtidos:
    - Acurácia no treino: 100.00%
    - Acurácia no teste: 100.00%
    - Convergiu na época: 2

3. Visualização:
    Iris [Result](./iris_result.png)

4. Análise:
   - O perceptron foi adequado? SIM
   - Razão: Classes linearmente separáveis
   - Resultado esperado: ~100% de acurácia

5. Pergunta:
- O que acontece se você usar Versicolor vs Virginica (classes 1 e 2)?

    - As classes 1 (Versicolor) e 2 (Virginica) NÃO são linearmente separáveis
    - O perceptron não conseguiria convergir (erro > 0 sempre)
    - Acurácia seria menor que 100%
    - Seria necessário um classificador mais complexo (ex: MLP, SVM)
    - Motivo: Sobreposição nas características das duas espécies

## 2. Moons Dataset  

1. Descrição do DatasetDescrição do Dataset:
    -  Formato: Duas 'luas' entrelaçadas
    - Separabilidade: NÃO-LINEAR
    - Ruído: 0.15 (torna o problema mais realista)
    - Amostras: 200 (balanceadas)

2. Limitações do Perceptron:
    - Só consegue aprender fronteiras LINEARES
    - Não pode separar dados em formato de 'lua'
    - Resultado: Acurácia baixa (~84%)
    - Nunca converge (erros > 0 sempre)

3. Resultados Obtidos:
    - Acurácia no treino: 80.71%
    - Acurácia no teste: 90.00%
    - Convergiu? NÃO
    - Erros finais: 32

4. Visualização:
    Moons [Result](./moons_result.png)

5. Análise:
    - Uma linha reta não pode separar duas luas
    - É necessária uma fronteira CURVA/NÃO-LINEAR
    - O perceptron só desenha LINHAS RETAS

6. Pergunta:
- Como você modificaria o algoritmo para resolver este problema?

- MULTI-LAYER PERCEPTRON (MLP/Rede Neural):
    - Adicionar camadas ocultas com funções de ativação não-lineares
    - Permite aprender fronteiras curvas complexas
    - Algoritmo: Backpropagation

- KERNEL TRICK (SVM):
    - Mapear dados para espaço de maior dimensão
    - Usar kernels RBF, polinomial, etc.
    - Fronteira não-linear no espaço original

- FEATURE ENGINEERING:
    - Criar features polinomiais: x1², x2², x1*x2
    - Adicionar termos não-lineares
    - Perceptron pode então usar essas features

- ENSEMBLE METHODS:
    - Random Forest, Gradient Boosting
    - Combinação de múltiplos classificadores
    - Maior flexibilidade para padrões complexos

- ALGORITMOS ESPECÍFICOS:
    - K-Nearest Neighbors (KNN)
    - Decision Trees
    - Naive Bayes com kernels