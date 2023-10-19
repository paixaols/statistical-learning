# Statistical Learning - statlearn

Statlearn (Statistical Learning) é um pacote escrito para fins didáticos. Os resultados obtidos são precisos, mas a performance de execução dos algorítmos não é otimizada. Em projetos reais utilize pacotes estabelecidos no mercado: sklearn, scipy etc.

## Dependências

- Python >= 3.9
- Numpy >= 1.23.5
- Pandas >= 2.0.3
- Scipy >= 1.9.3

## Modelos implementados

- classification
    - KNNClassifier
- cluster
    - KMeans
- linear_models
    - LinearRegressor

## Datasets

Alguns conjuntos de dados acompanham o código para facilitar testes. Os dados incluídos aqui são oriundos de outras fontes conhecidas.

### Iris: Seaborn

| sepal_length | sepal_width | petal_length | petal_width | species |
| ------------ | ----------- | ------------ | ----------- | ------- |
| float        | float       | float        | float       | string  |

Fonte: https://github.com/mwaskom/seaborn-data

### Advertising: An Introduction to Statistical Learning

| tv    | radio | newspaper | sales |
| ----- | ----- | --------- | ----- |
| float | float | float     | float |

Fonte: https://www.statlearning.com/resources-python