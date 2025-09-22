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

## 3. Breast Cancer Wisconsin

1. Descrição do Dataset:
    - Número de amostras: 569 (212 Maligno + 357 Benigno)
    - Número de features: 30 (características morfológicas dos núcleos celulares)
    - Classes: ['malignant' 'benign'] (maligno e benigno)
    - Distribuição das classes: desbalanceada (37% maligno, 63% benigno)
    - Tipo: Dataset médico real para diagnóstico de câncer de mama

2. Versões Implementadas:
    - **Versão A (2 features)**: mean radius + mean texture (para visualização)
    - **Versão B (30 features)**: todas as características disponíveis

3. Resultados Obtidos:

    | Versão | Features | Acurácia Treino | Acurácia Teste | Convergência |
    |--------|----------|-----------------|----------------|--------------|
    | A      | 2        | 87.19%          | 87.72%         | NÃO          |
    | B      | 30       | 98.99%          | 96.49%         | NÃO          |

4. Visualização:
    Breast Cancer [Result](./breast_cancer_result.png)

5. Análise Médica Crítica:

    **Matriz de Confusão (Versão B - 30 features):**
    ```
                    Predito
               Maligno  Benigno
    Real Maligno    59        5  ← Falsos Negativos (CRÍTICO!)
         Benigno     1      106  ← Falsos Positivos
    ```

    **Métricas Médicas:**
    - Sensibilidade (detectar malignos): 92.19%
    - Especificidade (detectar benignos): 99.07%
    - Taxa de Falsos Negativos: 7.81%
    - Taxa de Falsos Positivos: 0.93%

6. Comparação de Performance:
    - **Melhoria com 30 features**: +8.77% de acurácia
    - **Precision Maligno**: 98% vs 79% (Versão B vs A)
    - **F1-Score geral**: 0.96 vs 0.87
    - **Impacto dos Falsos Negativos**: 5 → 1 caso perdido

7. Análise:
    - **Adequação médica**: Acurácia excelente (96.49%)
    - **Problema crítico**: 1 caso maligno classificado como benigno
    - **Importância das features**: Mais features = diagnóstico muito melhor
    - **Limitação do Perceptron**: Não conseguiu convergir nem com 30 features

8. Impacto Médico:
    - **Falsos Negativos**: Pacientes com câncer seriam dispensados (FATAL)
    - **Falsos Positivos**: Pacientes saudáveis fariam exames desnecessários
    - **Prioridade**: Minimizar falsos negativos a qualquer custo

9. Recomendações:
    - Usar TODAS as features disponíveis (30 > 2)
    - Considerar algoritmos mais robustos (SVM, Random Forest, Deep Learning)
    - Implementar validação cruzada para maior confiabilidade
    - Otimizar para sensibilidade (detectar todos os casos malignos)
    - Sempre ter segunda opinião médica

10. Conclusão:
    - O perceptron mostrou potencial para diagnóstico médico
    - 30 features são cruciais para boa performance
    - Limitações lineares exigem algoritmos mais avançados
    - Para aplicação real: ensemble methods + validação rigorosa

## 4. Dataset de Classificação com Ruído

1. Descrição do Dataset:
    - Amostras: 200 (balanceadas)
    - Features: 2 (informativas, sem redundância)
    - Tipo: Dataset sintético controlado
    - Variáveis experimentais:
      - **class_sep**: 0.5 → 3.0 (separação entre classes)
      - **flip_y**: 0% → 20% (ruído nos rótulos)

2. Implementações Especiais:
    - **Perceptron com Early Stopping**: Paciência de 10 épocas
    - **Divisão tripla**: 60% treino, 20% validação, 20% teste
    - **Validação durante treinamento**: Para prevenir overfitting

3. Experimentos Realizados:

    **A. Variação da Separação (ruído fixo = 5%):**
    | Separação | Treino | Validação | Teste | Épocas | Convergência |
    |-----------|--------|-----------|-------|--------|--------------|
    | 0.5       | 66.7%  | 62.5%     | 70.0% | 21     | NÃO          |
    | 1.0       | 82.5%  | 72.5%     | 77.5% | 11     | NÃO          |
    | 1.5       | 92.5%  | 90.0%     | 92.5% | 18     | NÃO          |
    | 2.0       | 96.7%  | 95.0%     | 95.0% | 16     | NÃO          |
    | 2.5       | 99.2%  | 95.0%     | 95.0% | 12     | NÃO          |
    | 3.0       | 99.2%  | 97.5%     | 95.0% | 11     | NÃO          |

    **B. Variação do Ruído (separação fixa = 1.5):**
    | Ruído | Treino | Validação | Teste | Épocas | Convergência |
    |-------|--------|-----------|-------|--------|--------------|
    | 0%    | 94.2%  | 92.5%     | 95.0% | 11     | NÃO          |
    | 5%    | 92.5%  | 90.0%     | 92.5% | 18     | NÃO          |
    | 10%   | 87.5%  | 82.5%     | 95.0% | 13     | NÃO          |
    | 15%   | 80.8%  | 87.5%     | 80.0% | 12     | NÃO          |
    | 20%   | 80.8%  | 85.0%     | 75.0% | 11     | NÃO          |

4. Visualizações:
    - Ruído [Result](./ruido_result.png) - Regiões de decisão
    - Ruído [Analysis](./ruido_analysis.png) - Gráficos comparativos

5. Análise dos Resultados:

    **A. Impacto da Separação:**
    - **Melhoria dramática**: 70% → 95% (separação 0.5 → 2.0)
    - **Ponto de saturação**: A partir de separação 2.0, ganhos marginais
    - **Limite teórico**: ~95% devido aos 5% de ruído nos rótulos

    **B. Impacto do Ruído:**
    - **Degradação linear**: 95% → 75% (0% → 20% ruído)
    - **Limite teórico**: Performance máxima = 100% - ruído%
    - **Ruído crítico**: >15% torna perceptron inadequado

6. Early Stopping:
    - **Uso universal**: Todos os 11 experimentos utilizaram early stopping
    - **Benefícios**: Evita overfitting, treino mais eficiente
    - **Paciência ideal**: 10 épocas mostrou-se adequada

7. Análise:
    - **Sensibilidade ao ruído**: Perceptron muito sensível a rótulos incorretos
    - **Dependência da separação**: Crucial para boa performance
    - **Limitações lineares**: Fronteira reta inadequada para dados complexos
    - **Necessidade de validação**: Essential para dados ruidosos

8. Impacto Prático:
    - **Dados limpos** (ruído <5%, separação >2.0): Perceptron adequado
    - **Dados moderados** (ruído 5-10%, separação 1.0-2.0):️ Performance limitada
    - **Dados ruidosos** (ruído >15%, separação <1.0): Perceptron inadequado

9. Recomendações:

    Use Perceptron quando:
    - Separação entre classes ≥ 2.0
    - Ruído nos rótulos < 10%
    - Dataset linearmente separável
    - Velocidade é prioritária

    Considere alternativas quando:
    - Separação 1.0-2.0 com ruído 10-15%
    - Necessidade de maior robustez
    - Dados com outliers frequentes

     Evite Perceptron quando:
    - Ruído nos rótulos > 15%
    - Separação entre classes < 1.0
    - Fronteira de decisão complexa
    - Alta sobreposição entre classes

10. Conclusão:
    - Early stopping é **essencial** para dados ruidosos
    - Separação entre classes é **mais crítica** que o nível de ruído
    - Para aplicações reais: **SVM**, **Random Forest** ou **Ensemble Methods**
    - Perceptron ideal apenas para **dados bem separados e limpos**