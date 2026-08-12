<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/cover.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/cover_light.png">
  <img alt="Redes Neurais e Aprendizado Profundo" src="/.github/cover_light.png">
</picture>

# Redes Neurais e Aprendizado Profundo

Repositório com notas de aula, tarefas e materiais de estudo da disciplina de **Redes Neurais e Aprendizado Profundo** do curso de Ciência da Computação da Universidade Federal de Alagoas (UFAL).

## 📋 Estrutura do Repositório

### 🧪 `/tasks` - Tarefas Práticas

Exercícios de implementação com código incompleto a ser preenchido:

- **task-01-simple-perceptron** - Implementação do algoritmo de aprendizado do Perceptron do zero, com dados sintéticos (`make_blobs`) e visualização da fronteira de decisão
- **task-02-halfmoon-perceptron** - Aplicação do Perceptron a dados não linearmente separáveis (formato "halfmoon"), demonstrando as limitações de classificadores lineares

### 📓 `/lecture-notes` - Notas de Aula

Slides, PDFs e materiais complementares das aulas, organizados por capítulo:

- **Ch01-Deep-Learning-Applications** - Aplicações de Aprendizado Profundo
- **Ch02-Perceptron** - História, convergência e implementação do Perceptron (`simple_perceptron.py`)
- **Ch03-Linear-Regression** - Regressão Linear (PDF + slides HTML)
- **Ch04-Logistic-Regression** - Regressão Logística (PDF + slides HTML)
- **Ch05-NN-Overview** - Visão geral de Redes Neurais
- **Ch06-Activation-Functions** - Funções de Ativação
- **ch06-teaching-dl-to-generalize-5** - Generalização em Aprendizado Profundo
- **ch07-convolutional-neural-networks** - Redes Neurais Convolucionais (ConvNets)

### 📖 `/references` - Livros de Referência

<details>
<summary>Livros-texto clássicos e atuais sobre redes neurais e aprendizado profundo:</summary>

- _Deep Learning_ - Ian Goodfellow, Yoshua Bengio e Aaron Courville (MIT Press, 2016)
- _Neural Networks and Deep Learning: A Textbook_ - Charu C. Aggarwal (Springer, 2023)
- _Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow_ - Aurélien Géron (2ª edição)
- _Deep Learning with Python_ (3ª edição)

</details>

## 🚀 Como Executar

As tarefas utilizam `numpy`, `matplotlib` e `scikit-learn`. Após completar as partes indicadas nos arquivos, execute:

```bash
python tasks/task-01-simple-perceptron/task-01-simple-perceptron.py
python tasks/task-02-halfmoon-perceptron/task-02-halfmoon-perceptron.py
```
