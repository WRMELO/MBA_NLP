
# 📂 README — Projeto NLP Classificação de Reclamações

### 🎯 **Objetivo**
Este diretório armazena **todos os artefatos finais** do pipeline supervisionado de classificação de reclamações, utilizando múltiplas abordagens de vetorização e feature engineering.

---

## 📊 **Artefatos contidos**

| Artefato                                   | Descrição                                                                                      |
|--------------------------------------------|------------------------------------------------------------------------------------------------|
| `X_train_*`, `X_test_*`                    | Matrizes vetorizadas para treino e teste (extensões `.npz` para esparsas e `.npy` para densas) |
| `y_train_*`, `y_test_*`                    | Rótulos estratificados correspondentes, salvos como `.csv`                                     |
| `relatorio_comparativo_classificacao.csv`  | Comparação de métricas supervisionadas (Accuracy, Precision, Recall, F1 Score Weighted)        |
| `grafico_comparativo_f1score.png`          | Gráfico de barras com linha de benchmark F1 ≥ 75%                                              |
| `README.md`                                | Documento rastreável para versionamento                                                        |

---

## ✅ **Baseline selecionado**

Após análise comparativa:
- **Abordagem escolhida:** **Bag of Words (BoW)** + **Logistic Regression**
- **F1 Score Weighted atingido:** **0.9005**
- **Justificativa:** Simplicidade e interpretabilidade, superando abordagens mais complexas como Word2Vec e Sentence-Transformer para o dataset em questão.

---

## 🗂️ **Rastreamento**

- Todos os vetores seguem padronização de nomes coerente com o notebook principal.
- Versões de bibliotecas utilizadas:  
  - `gensim` para Word2Vec pré-treinado  
  - `sentence-transformers==3.2.1`  
  - `transformers==4.46.3`  
  - `numpy>=1.24` para embeddings densos.

---

