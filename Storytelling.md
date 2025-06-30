
# 🗺️ **Resumo Detalhado — Pipeline NLP até Etapa 2.2**

Este documento registra de forma clara e rastreável **todas as etapas já concluídas** no projeto de classificação de chamados, seguindo o **Plano de Ação Detalhado** e o **PROTOCOLO LLM UNIVERSAL v5.2**.

---

## ✅ **1️⃣ Setup de Infraestrutura**

**Objetivo:** Garantir que o ambiente tenha todas as bibliotecas, modelos de linguagem e variáveis de caminho necessários para execução reproducível.

**Ações realizadas:**
- **Instalação e verificação de bibliotecas:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy`, `unidecode`, `tqdm` e `sentence-transformers` foram validados ou instalados.
- **Download do modelo SpaCy `pt_core_news_sm`:** confirmamos que o modelo está disponível para tarefas de tokenização e lematização em português.
- **Instalação de `punkt` e `stopwords` do NLTK:** tentativa de uso para tokenização inicial, alinhado às práticas de introdução.
- **Definição de constantes globais:** `URL_ORIGINAL`, `BASE_DIR` e `PATH_RAW` configurados para rastrear a origem dos dados (`S3`) e o armazenamento local no Google Drive (`/MBA_NLP/bases_criadas`).

---

## ✅ **2️⃣ EDA Inicial**

**Objetivo:** Entender a estrutura do dataset, verificar nulos, analisar distribuição de textos e categorias.

**Ações realizadas:**
- **Carregamento do dataset original:** `dados_originais.csv` foi carregado de forma validada via `PATH_RAW`.
- **Exploração de metadados:** Usamos `df.info()`, `df.shape` e `df.head(20)` para verificar consistência.
- **Análise de valores nulos:** Percentuais verificados — sem inconsistências graves.
- **Análise de comprimento dos textos:** Histogramas indicaram variação de textos de 20 a 2.000+ caracteres.
- **Balanceamento de classes:** `value_counts()` + gráfico de barras revelaram leve desbalanceamento entre categorias como `Serviços de Conta Bancária` e `Outros`.

---

## ✅ **3️⃣ Limpeza de Texto — Primeira Fase**

**Objetivo:** Remover ruídos e anonimizar dados sensíveis presentes em `descricao_reclamacao`.

**Ações realizadas:**
- **Função `clean_text()`:** Aplicou lowercase, remoção de acentos (`unidecode`), pontuação e espaços extras.
- **Identificação de `xxxx`:** Reconhecemos que os placeholders representam informações sensíveis como PII, datas e IDs.
- **Função `replace_xxxx_tokens()`:** Substituímos padrões `xxxx` por tokens semânticos `<PII>`, `<DATE>`, `<ID>` ou `<UNK>`.
- **Salvamento da versão intermediária:** `dados_com_tokens.csv` armazenado em `/bases_criadas`.

---

## ✅ **4️⃣ Tokenização — Estratégia Revisada**

**Contexto:** A abordagem inicial usava `NLTK` com `punkt`. Devido a falhas recorrentes no ambiente Colab (bug `punkt_tab not found`), foi decidida a **troca metodológica**:

- **Fontes acadêmicas e materiais de aula recomendam SpaCy** para tokenização em português, pela robustez do modelo `pt_core_news_sm`.
- A tokenização SpaCy é linguística, com suporte nativo ao idioma, incluindo POS-Tagging e lematização.
- Adicionamos o filtro `token.is_alpha` para extrair apenas palavras válidas, alinhado aos exemplos `Aula_1_DTS_PLN_Exercícios`.

**Resultado:**  
- A coluna `texto_tokens_list` foi criada com tokens extraídos do SpaCy.
- Utilizamos `tqdm` integrado (`tqdm.pandas()`) para monitorar o progresso da tokenização, reforçando rastreabilidade acadêmica.
- O arquivo final `dados_tokens_tokenized.csv` foi salvo na pasta `/bases_criadas`.

---

## 🗃️ **5️⃣ Estratégia com NotebookLM**

**Decisão metodológica:**  
Para **acompanhar, validar e documentar** cada etapa do fluxo, optamos por:
- **Executar o pipeline no Google Colab**, integrando o Google Drive para persistência.
- **Registrar cada célula explicativa em Markdown**, no padrão exigido pelo `PROTOCOLO LLM UNIVERSAL v5.2`.
- **Validar iterações complexas (como a substituição de placeholders e refinamentos linguísticos)** usando o **NotebookLM**, que atua como repositório de experimentos e como ponte de controle para feedback de boas práticas.

Assim, garantimos **rastreabilidade total**, versionamento dos dados (`dados_originais.csv`, `dados_com_tokens.csv`, `dados_tokens_tokenized.csv`) e alinhamento com a trilha acadêmica.

---

## 🚩 **Resumo do Progresso**

| Macro-Bloco | Status |
|-------------|--------|
| Setup Infraestrutura | ✅ Concluído |
| EDA Inicial | ✅ Concluído |
| Limpeza de Texto | ✅ Fase Base Concluída |
| Tokenização | ✅ Concluída com SpaCy + `tqdm` |
| Stopwords | ⏭️ Próxima etapa |

---

### 📌 **Próximo Passo**

A seguir, avançaremos para a **Etapa 2.3 — Filtro de Stopwords (pt)**, aplicando a mesma robustez técnica, combinando NLTK e SpaCy, para manter o texto mais limpo para vetorização.

---
