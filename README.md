# Atividade Prática de Inteligência Artificial

## Dupla

- Leandro Carvalho
- Yuri Carneiro

## 1. Iris

1. Descrição do DatasetDescrição do Dataset
    - Número de amostras: 100 (50 Setosa + 50 Versicolor)
    - Número de features: 2 (comprimento sépala e pétala)
    - Distribuição das classes: balanceada (50/50)
    - É linearmente separável? SIM (classes 0 e 1 são separáveis)

2. Resultados
    - Acurácia no treino: 100.00%
    - Acurácia no teste: 100.00%
    - Convergiu na época: 2

3. VisualizaçõesVisualizações
    [Iris Result](./iris_result.png)

4. Análise
   - O perceptron foi adequado? SIM
   - Razão: Classes linearmente separáveis
   - Resultado esperado: ~100% de acurácia

5. Pergunta
    - As classes 1 (Versicolor) e 2 (Virginica) NÃO são linearmente separáveis
    - O perceptron não conseguiria convergir (erro > 0 sempre)
    - Acurácia seria menor que 100%
    - Seria necessário um classificador mais complexo (ex: MLP, SVM)
    - Motivo: Sobreposição nas características das duas espécies

## 2. 