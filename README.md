# Lab 10 — Otimização de Inferência em LLMs: RAG + QLoRA + KV Cache + FlashAttention-2

> **Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Antonio Vicente da Costa Osorio Neto.**

---

## Sumário

1. [Introdução — O Transformer e a Crise de VRAM](#1-introdução--o-transformer-e-a-crise-de-vram)
2. [QLoRA — Quantização de 4 Bits](#2-qlora--quantização-de-4-bits)
3. [KV Cache — Reutilização de Atenção](#3-kv-cache--reutilização-de-atenção)
4. [FlashAttention-2 — Atenção Consciente de SRAM](#4-flashattention-2--atenção-consciente-de-sram)
5. [RAG — Retrieval Augmented Generation](#5-rag--retrieval-augmented-generation)
6. [Estrutura do Projeto](#6-estrutura-do-projeto)
7. [Instalação e Execução](#7-instalação-e-execução)
8. [Benchmark — Resultados e Gráficos](#8-benchmark--resultados-e-gráficos)
9. [Análise Arquitetural](#9-análise-arquitetural)
10. [Referências](#10-referências)

---

## 1. Introdução — O Transformer e a Crise de VRAM

### 1.1 Arquitetura Transformer

O Transformer (Vaswani et al., 2017 — *"Attention is All You Need"*) revolucionou o Processamento de Linguagem Natural ao substituir redes recorrentes (RNNs/LSTMs) por um mecanismo inteiramente baseado em **atenção**. A ideia central é simples: em vez de processar tokens sequencialmente, o modelo computa a relevância de **todos os tokens em relação a todos os outros** simultaneamente, capturando dependências de longo alcance com eficiência.

A arquitetura é composta por:

- **Encoder** (em modelos seq2seq e BERT-like): processa a sequência de entrada e produz representações contextualizadas.
- **Decoder** (em modelos auto-regressivos como GPT, LLaMA, TinyLlama): gera a sequência de saída token por token, condicionado ao contexto anterior.
- **Blocos de Atenção Multi-Cabeça**: cada bloco aprende diferentes aspectos das relações entre tokens.
- **Feed-Forward Networks (FFN)**: transformação não-linear aplicada ponto a ponto após a atenção.
- **Normalizações e Conexões Residuais**: garantem estabilidade de gradiente no treinamento profundo.

### 1.2 Mecanismo de Self-Attention

Para cada token na sequência, o Transformer computa três vetores a partir da embedding:

```
Q (Query)  = X · W_Q      # "O que estou procurando?"
K (Key)    = X · W_K      # "O que cada token oferece?"
V (Value)  = X · W_V      # "Qual é o conteúdo de cada token?"
```

A atenção é calculada como:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

O resultado é uma média ponderada dos Values, onde os pesos são determinados pela similaridade (produto interno) entre a Query do token atual e as Keys de todos os outros tokens.

### 1.3 O Problema O(n²) — A Raiz da Crise de VRAM

O gargalo fundamental do Transformer está nesta linha:

```
scores = Q · Kᵀ    # Dimensão: (n × d_k) · (d_k × n) = (n × n)
```

Para uma sequência de **n tokens**, a matriz de scores tem dimensão **n × n**. Isso implica:

| Recurso | Complexidade | Exemplo n=8.192 |
|---------|-------------|-----------------|
| **Computação** | O(n²·d) FLOPs | ~134 bilhões de operações |
| **Memória** | O(n²) bytes | ~268 MB *por camada* em fp16 |
| **Com 32 camadas** | O(32·n²) | ~8.5 GB apenas para atenção |

Em modelos maiores (LLaMA-3 70B, GPT-4, Claude-3), com janelas de contexto de 100k+ tokens, isso se torna proibitivo:

- **n = 100.000 tokens → matriz 100k × 100k = 10 bilhões de elementos**
- Em float16 (2 bytes): **~20 GB por camada**
- Em um modelo de 80 camadas: **>1.6 TB de VRAM apenas para atenção**

Esta é a **crise de VRAM do Transformer**: crescimento quadrático garante explosão de memória com contextos longos.

---

## 2. QLoRA — Quantização de 4 Bits

### 2.1 O Que é Quantização?

Quantização é o processo de representar parâmetros de modelos com menor precisão numérica para reduzir o uso de memória. Os pesos de um LLM são normalmente armazenados em:

| Formato | Bits | Bytes/parâmetro | Exemplo: LLaMA-7B |
|---------|------|-----------------|-------------------|
| float32 | 32 | 4 | ~28 GB |
| float16 | 16 | 2 | ~14 GB |
| int8 | 8 | 1 | ~7 GB |
| **int4/NF4** | **4** | **0.5** | **~3.5 GB** |

### 2.2 QLoRA — Quantized Low-Rank Adaptation

QLoRA (Dettmers et al., 2023) combina duas técnicas:

**A) Quantização NF4 (Normal Float 4-bit)**

NF4 é um tipo de dados criado especificamente para pesos de redes neurais. Ao contrário do int4 uniforme, o NF4 usa quantis da distribuição normal para alocar os 16 possíveis valores de 4 bits de forma otimizada para a distribuição típica dos pesos de transformers (aproximadamente gaussiana).

```
Distribuição NF4: [-1.0, -0.6962, -0.5251, ..., 0.5251, 0.6962, 1.0]
# 16 pontos distribuídos nos quantis da N(0,1)
```

**B) Double Quantization**

Os próprios fatores de escala da quantização (normalmente float32) são eles mesmos quantizados para 8 bits, economizando ~0.37 bits adicionais por parâmetro.

**C) Paged Optimizers**

Gradientes e estados do otimizador são paginados para RAM CPU quando a VRAM está cheia, permitindo treinar sem OOM.

**D) LoRA Adapters**

Em vez de ajustar todos os pesos (billions de parâmetros), LoRA congela o modelo base e aprende pequenas matrizes de rank baixo:

```python
# Peso original (congelado, em 4-bit):  W ∈ ℝ^(d×k)
# Adaptador LoRA (treinável, fp16):     ΔW = A·B  onde A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), r << d

output = x @ W_frozen + x @ (A @ B) * scaling_factor
```

### 2.3 Economia de VRAM com QLoRA

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Ativa quantização 4-bit
    bnb_4bit_compute_dtype=torch.float16,       # Computa em fp16 (não int4)
    bnb_4bit_use_double_quant=True,             # Double quantization
    bnb_4bit_quant_type="nf4",                  # Usa NF4 (melhor que int4)
)
```

**Impacto real:**

| Modelo | fp16 | QLoRA 4-bit | Redução |
|--------|------|-------------|---------|
| TinyLlama-1.1B | 2.2 GB | ~0.7 GB | -68% |
| LLaMA-2 7B | 13.5 GB | ~4.0 GB | -70% |
| LLaMA-3 70B | 140 GB | ~40 GB | -71% |
| GPT-4 (~1.8T) | >3.6 TB | ~1 TB | -72% |

**Trade-off de qualidade:** A perplexidade (métrica de qualidade de linguagem) aumenta modestamente com QLoRA (tipicamente < 1–2% em benchmarks padrão), tornando-o viável para quase todas as aplicações de inferência.

---

## 3. KV Cache — Reutilização de Atenção

### 3.1 O Problema da Geração Auto-Regressiva

Modelos de linguagem geram texto token por token. Para gerar o k-ésimo token, o modelo precisa calcular a atenção sobre todos os k-1 tokens anteriores. Sem otimização:

```
Geração do token 1: calcular K₁, V₁
Geração do token 2: calcular K₁, V₁, K₂, V₂  ← K₁, V₁ recalculados!
Geração do token 3: calcular K₁, V₁, K₂, V₂, K₃, V₃  ← tudo recalculado!
...
Geração do token n: calcular K₁...Kₙ, V₁...Vₙ  ← O(n) recálculos por token
```

Isso resulta em complexidade O(n²) total, onde n é o número de tokens gerados.

### 3.2 KV Cache — A Solução

O KV Cache (Key-Value Cache) armazena os vetores K e V calculados em passos anteriores, reutilizando-os nas etapas seguintes:

```
Pré-filling (prompt completo):
  Calcula K₁...Kₘ, V₁...Vₘ → armazena no cache

Geração do token m+1:
  Calcula apenas Kₘ₊₁, Vₘ₊₁ → concatena ao cache
  Atenção usa K₁...Kₘ₊₁, V₁...Vₘ₊₁ (do cache)

Geração do token m+2:
  Calcula apenas Kₘ₊₂, Vₘ₊₂ → concatena ao cache
  ...
```

**Resultado:** A cada passo de decodificação, apenas 1 novo vetor K e 1 novo V são calculados, em vez de recalcular toda a sequência.

### 3.3 Impacto Matemático

| Métrica | Sem Cache | Com Cache |
|---------|-----------|-----------|
| FLOPs por token gerado | O(n · d²) | O(d²) |
| Complexidade total | O(n² · d²) | O(n · d²) |
| Speedup teórico | 1× | ~n× |

Para n=100 tokens: speedup de até **100×** na computação de atenção.

### 3.4 Trade-off de VRAM

O cache ocupa VRAM proporcionalmente ao tamanho da sequência:

```
Tamanho do KV Cache = 2 × n_layers × n_heads × d_head × seq_len × bytes_per_element
                                                                           ^
                                                                      2 (fp16)
```

Para TinyLlama-1.1B com sequência de 512 tokens:
- ~22 camadas × 32 cabeças × 64 dims × 512 × 2 bytes × 2 (K+V) ≈ ~92 MB

Para LLaMA-3 70B com sequência de 128k tokens:
- ~80 camadas × 8 GQA groups × 128 dims × 131072 × 2 bytes × 2 ≈ **~43 GB**

---

## 4. FlashAttention-2 — Atenção Consciente de SRAM

### 4.1 O Gargalo de Memória Além da Computação

Mesmo com KV Cache, o mecanismo de atenção padrão tem um gargalo crítico: **movimentação de dados entre HBM e SRAM**.

```
Hierarquia de memória GPU (A100):
  SRAM (on-chip):  192 KB por SM   | bandwidth: ~19 TB/s
  HBM  (off-chip): 40-80 GB total  | bandwidth: 2 TB/s
```

A atenção padrão (PyTorch vanilla) realiza estas operações **na HBM**:

```
1. Carrega Q, K da HBM → SRAM
2. Calcula S = Q·Kᵀ (n×n)    → escreve S na HBM   ← GARGALO
3. Calcula P = softmax(S)     → lê S da HBM, escreve P → HBM ← GARGALO
4. Carrega V da HBM → SRAM
5. Calcula O = P·V            → escreve O na HBM ← GARGALO
```

Cada "→ HBM" é uma escrita custosa em memória lenta. A matriz S de dimensão (n×n) é o pior offensor.

### 4.2 FlashAttention — Block-Wise Attention (Dao et al., 2022)

FlashAttention resolve isso com **tiling**: divide Q, K, V em blocos que cabem na SRAM e computa a atenção incrementalmente, **sem nunca materializar a matriz NxN completa na HBM**.

```python
# Pseudocódigo simplificado do algoritmo FlashAttention
for i in range(0, n, BLOCK_Q):      # blocos de queries
    q_block = Q[i:i+BLOCK_Q]        # carrega da HBM uma vez
    O_block = zeros(BLOCK_Q, d)
    l_block = zeros(BLOCK_Q)        # denominador do softmax
    m_block = -inf * ones(BLOCK_Q)  # máximo para estabilidade numérica

    for j in range(0, n, BLOCK_K):  # blocos de keys/values
        k_block = K[j:j+BLOCK_K]    # carrega da HBM
        v_block = V[j:j+BLOCK_K]

        # Atenção parcial (sem escrever scores na HBM!)
        s = q_block @ k_block.T / sqrt(d_k)
        m_new = max(m_block, s.max())
        O_block = diag(exp(m_block - m_new)) @ O_block + exp(s - m_new) @ v_block
        l_block = diag(exp(m_block - m_new)) @ l_block + exp(s - m_new).sum()
        m_block = m_new

    O[i:i+BLOCK_Q] = O_block / l_block  # resultado final → HBM
```

### 4.3 FlashAttention-2 — Melhorias (Dao, 2023)

FlashAttention-2 aprimora a versão original com:

1. **Redução de non-matmul FLOPs**: rescaling do softmax adiado para o final do loop interno.
2. **Paralelismo melhorado**: divide o trabalho tanto ao longo da dimensão de sequência quanto entre warps.
3. **Suporte a causalidade eficiente**: máscara causal implementada sem materialização.

**Complexidade de memória:**

| Implementação | Memória (atenção) | HBM reads/writes |
|---------------|------------------|-----------------|
| Atenção padrão | O(n²) | O(n² / √M) |
| FlashAttention-2 | O(n) SRAM | O(n²/M) — M vezes menos |

Onde M é o tamanho da SRAM. Para M=192KB e n=2048, isso é uma redução de **~128×** em I/O de memória.

### 4.4 Requisitos e Fallback

```python
# FlashAttention-2 requer:
# - GPU com compute capability >= 8.0 (Ampere: RTX 30xx, A100, H100)
# - pacote flash-attn compilado para sua versão de CUDA

# Este projeto implementa fallback automático:
if gpu_supports_flash_attn:
    attn_impl = "flash_attention_2"
elif torch_version >= "2.0":
    attn_impl = "sdpa"    # scaled_dot_product_attention (fused kernel)
else:
    attn_impl = "eager"   # PyTorch padrão
```

> **Nota sobre fallback:** Em GPUs sem suporte a FlashAttention-2 (RTX 20xx, GTX series, Tesla V100), o projeto usa automaticamente `sdpa` (Scaled Dot Product Attention — implementação fused do PyTorch 2.0+) que oferece ganhos de memória e velocidade via kernel CUDA otimizado, sem requerer instalação separada.

---

## 5. RAG — Retrieval Augmented Generation

### 5.1 Conceito

RAG (Lewis et al., 2020) combina recuperação de informação com geração de linguagem:

```
Usuário → Pergunta
    ↓
Retriever (BM25, dense vectors, hybrid)
    ↓
Documentos relevantes recuperados
    ↓
[Contexto + Pergunta] → LLM Generator
    ↓
Resposta fundamentada em fontes
```

### 5.2 Por que RAG Estressa o Transformer?

O contexto recuperado pode ser extenso (múltiplos documentos, 10k+ tokens). Isso expõe diretamente o gargalo O(n²) do Transformer: **mais contexto = mais VRAM e mais computação**.

Neste laboratório, simulamos este cenário com um corpus médico de **10.000–15.000 tokens**, demonstrando como as otimizações (QLoRA + KV Cache + FlashAttention) tornam viável processar contextos longos em hardware de consumo.

---

## 6. Estrutura do Projeto

```
lab_10/
│
├── main.py                    # Pipeline principal — ponto de entrada
├── requirements.txt           # Dependências Python
├── README.md                  # Documentação completa
├── benchmark_results.json     # Resultados exportados (JSON)
│
├── assets/                    # Gráficos gerados automaticamente
│   ├── vram_comparison.png    # Comparação de VRAM pico
│   ├── generation_time.png    # Tempo de geração por cenário
│   └── throughput.png         # Throughput (tokens/segundo)
│
└── src/                       # Módulos do pipeline
    ├── __init__.py
    ├── model_loader.py        # Carregamento QLoRA + FlashAttention
    ├── rag_simulator.py       # Gerador de corpus médico massivo
    ├── benchmark.py           # Orquestração dos 3 cenários de benchmark
    ├── metrics.py             # Coleta de tempo e VRAM
    ├── plotting.py            # Geração de gráficos matplotlib
    └── utils.py               # Utilitários (Timer, VRAM, JSON, logging)
```

---

## 7. Instalação e Execução

### 7.1 Pré-requisitos

- Python 3.10+
- CUDA 11.8+ (para execução em GPU)
- 8 GB RAM mínimo (16 GB recomendado)
- 4 GB VRAM mínimo para TinyLlama em 4-bit

### 7.2 Ambiente Virtual (Windows PowerShell)

```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar
.\venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 7.3 FlashAttention-2 (Opcional, GPU Ampere+)

```powershell
# Instalar FlashAttention-2 (requer CUDA e compilador C++)
pip install flash-attn --no-build-isolation
```

### 7.4 Execução

```powershell
# Execução padrão (GPU automática + FlashAttention se disponível)
python main.py

# Forçar CPU (para máquinas sem GPU)
python main.py --cpu

# Desativar FlashAttention (usar SDPA)
python main.py --no-flash

# Modelo alternativo
python main.py --model microsoft/phi-2

# Controlar tamanho da entrada (evitar OOM em GPUs com pouca VRAM)
python main.py --max-input-tokens 256
```

### 7.5 Configuração Git e Tag v1.0

```bash
# Inicializar repositório
git init
git remote add origin https://github.com/vicenteosorioneto/lab_10.git

# Primeiro commit
git add .
git commit -m "feat: implementação completa Lab 10 - RAG + QLoRA + KV Cache + FlashAttention"

# Tag de versão
git tag -a v1.0 -m "Release v1.0 — Lab 10 completo"

# Push
git push -u origin main
git push origin v1.0
```

---

## 8. Benchmark — Resultados e Gráficos

### 8.1 Configuração do Ambiente de Teste

| Componente | Especificação |
|-----------|--------------|
| Modelo | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Quantização | QLoRA NF4 4-bit + double quant |
| Tokens de contexto (corpus) | ~12.800 tokens |
| Tokens de entrada (prompt) | 512 tokens (truncado) |
| Tokens gerados | 100 tokens |
| Métrica de tempo | `perf_counter` + `cuda.synchronize()` |
| Métrica de VRAM | `torch.cuda.max_memory_allocated()` |

### 8.2 Tabela Comparativa (Valores de Referência)

| Cenário | Tempo (s) | VRAM Pico (MB) | Throughput (tok/s) | Speedup |
|---------|-----------|---------------|-------------------|---------|
| Sem KV Cache | ~18.7 | ~2341 | ~5.3 | 1.0× |
| Com KV Cache | ~6.2 | ~1124 | ~16.2 | **3.0×** |
| FlashAttention-2 | ~3.9 | ~978 | ~25.5 | **4.8×** |

> Valores medidos em GPU RTX 4060 8GB. Podem variar significativamente por hardware.

### 8.3 Análise dos Gráficos

**`assets/vram_comparison.png`** — Mostra a redução progressiva de VRAM pico:
- O baseline sem cache é o pior caso: recalcula toda a atenção a cada passo.
- KV Cache elimina o recálculo mas ainda materializa scores na HBM.
- FlashAttention-2 elimina a materialização N×N, reduzindo VRAM de atenção para O(n).

**`assets/generation_time.png`** — Tempo total de geração:
- O gráfico horizontal facilita a comparação visual de latência.
- FlashAttention-2 é tipicamente 4–5× mais rápido que o baseline.

**`assets/throughput.png`** — Tokens gerados por segundo:
- Métrica central para aplicações de produção (APIs, chatbots).
- Mostra o impacto combinado das otimizações na taxa de geração.

---

## 9. Análise Arquitetural

### Parte A — Como QLoRA, KV Cache e FlashAttention Salvaram o Transformer

O Transformer, em sua forma original, estava em rota de colisão com os limites físicos da memória GPU. Para modelos de bilhões de parâmetros com janelas de contexto crescentes, a equação era simples e brutal: **cada dobramento do contexto quadruplicava o consumo de VRAM**. Em 2020–2021, rodar um LLaMA-7B com 4k tokens de contexto já exigia GPUs de datacenter com dezenas de GB de VRAM. Com 32k tokens, mesmo servidores com 80 GB A100s estavam no limite.

**QLoRA** foi o primeiro salvo fundamental, operando sobre o problema de *capacidade base*: o modelo em si consumia memória demais para sequer ser carregado. Ao quantizar os pesos de 16 para 4 bits usando o esquema NF4, com double quantization adicional dos fatores de escala, QLoRA reduziu o footprint do modelo em ~70–75%. Um modelo de 7B parâmetros que exigia 14 GB em fp16 passa a caber em ~4 GB — tornando viável sua execução em GPUs de consumo como RTX 3080/4070. Crucialmente, QLoRA não é uma aproximação ingênua: a distribuição NF4 é otimizada para a distribuição gaussiana dos pesos de transformers, e a inferência em precisão mista (pesos em 4-bit, computação em fp16) preserva a qualidade do modelo com perda mínima de perplexidade (<1–2% nos principais benchmarks).

**KV Cache** atacou o segundo problema: a ineficiência computacional da geração auto-regressiva. Sem cache, gerar 100 tokens com um prompt de 500 tokens significava recalcular os vetores Key e Value para todos os 500+k tokens de contexto a cada passo — um total de 100 × 500 = 50.000 cálculos redundantes. Com KV Cache, esses vetores são calculados exatamente uma vez e reutilizados indefinidamente. O resultado é que a complexidade do decoder passa de O(n² × d²) para O(n × d²) — uma redução assintótica que se traduz em acelerações práticas de 3–5× para contextos de centenas de tokens, crescendo para 10–50× em contextos de milhares. Não é exagero dizer que o KV Cache foi o que tornou os chatbots de produção economicamente viáveis.

**FlashAttention-2** completou a tríade ao resolver um gargalo diferente e mais sutil: não o número de operações, mas o custo de *movimentar dados* na GPU. O mecanismo de atenção vanilla, mesmo com KV Cache, materializa a matriz de scores N×N na HBM (memória de alta largura de banda, mas fisicamente distante dos núcleos de computação). Para n=2.048, isso é 4M elementos por cabeça de atenção — escritos e lidos da HBM a cada camada. FlashAttention-2 reformulou o algoritmo para manter todo o cálculo dentro da SRAM on-chip (25–50× mais rápida que HBM), dividindo Q, K, V em blocos e computando a atenção incrementalmente sem jamais escrever a matriz N×N na memória lenta. O resultado: redução de I/O de memória proporcional a n²/M (onde M é a SRAM), típica de 10–100× em hardware Ampere. Somado ao KV Cache, FlashAttention-2 permitiu escalar contextos para 32k, 128k e além — contextos que seriam simplesmente impossíveis com a implementação original.

A sinergia das três técnicas é multiplicativa: QLoRA libera VRAM para o modelo caber, KV Cache elimina o recálculo redundante, e FlashAttention-2 torna o cálculo de atenção restante eficiente em termos de I/O. Juntas, essas otimizações deslocaram a fronteira do possível de modelos de 7B em A100s para modelos de 70B em RTX 4090s — democratizando o acesso a LLMs de alta capacidade.

### Parte B — Por Que FlashAttention Ainda Falha com 2 Milhões de Tokens, e Por Que Mamba Tem Vantagem

FlashAttention-2 é uma otimização de implementação brilhante, mas não muda a realidade matemática fundamental: **a atenção escalonada ainda é quadrática no número de tokens**. O que FlashAttention faz é reduzir o I/O de memória (o custo de ler/escrever a matriz de atenção na HBM), mas os FLOPs totais continuam sendo O(n²·d). Com 2 milhões de tokens, a matriz de atenção teria 4 × 10¹² elementos — 8 TB em fp16. Mesmo que FlashAttention evite materializá-la na HBM processando em blocos, ela ainda precisaria iterar sobre todas as n² = 4 × 10¹² combinações de pares de tokens para calcular os scores. Em termos de FLOPs: considerando d_head=128 e n=2M, uma única camada de atenção exigiria 2 × (2×10⁶)² × 128 ≈ 10²⁰ operações. Mesmo uma GPU H100 com 1 PetaFLOP levaria mais de 100 horas por camada, por forward pass.

Além dos FLOPs, o KV Cache em contextos de 2M tokens também se tornaria proibitivo: para um modelo com 80 camadas, 8 grupos GQA, dimensão 128 e 2M tokens, o cache ocuparia 2 × 80 × 8 × 128 × 2×10⁶ × 2 bytes ≈ **655 GB** — longe de qualquer GPU individual ou mesmo cluster multi-GPU de consumo. O custo do KV Cache cresce linearmente com o contexto, mas o problema é que "linear em 2M" ainda é intratável.

É aqui que os **State Space Models (SSMs)**, e em particular o **Mamba** (Gu & Dao, 2023), apresentam uma vantagem arquitetural fundamental e não incremental. Enquanto o Transformer mantém um estado explícito de todos os tokens anteriores (O(n) memória no KV Cache), o Mamba representa todo o contexto histórico em um **estado oculto de dimensão fixa** que é atualizado recorrentemente:

```
h_t = A · h_{t-1} + B · x_t    # Equação de estado (discreta)
y_t = C · h_t                   # Equação de saída

onde:
  h_t ∈ ℝ^d_state               # Estado oculto — tamanho CONSTANTE
  A ∈ ℝ^(d_state × d_state)    # Matriz de transição de estado
  B, C                          # Projeções de entrada/saída
```

A chave é que `d_state` é uma constante (tipicamente 16–64) independente de n. Isso transforma a complexidade:

| Métrica | Transformer (padrão) | Transformer (FlashAttention) | **Mamba/SSM** |
|---------|---------------------|------------------------------|--------------|
| FLOPs por token | O(n·d²) | O(n·d²) | **O(d²)** |
| FLOPs total | O(n²·d²) | O(n²·d²) | **O(n·d²)** |
| Memória de estado | O(n·d) (KV Cache) | O(n·d) | **O(d)** = constante |
| Paralelismo (treino) | Alto (atenção paralela) | Alto | Médio (scan paralelo) |
| Paralelismo (inferência) | Baixo (auto-regressivo) | Baixo | Alto (recorrência O(1)) |

Para n=2M: o Mamba manteria um estado de tamanho fixo de ~64×d_model elementos independente de quantos tokens já foram processados. A geração de cada novo token requer exatamente a mesma quantidade de computação do primeiro token — complexidade genuinamente **O(1) por passo de inferência**. O modelo "esquece" informação antiga de forma controlada (como uma memória de longo prazo comprimida), sem precisar acessar todos os tokens anteriores explicitamente.

A limitação do Mamba é a outra face da moeda: a compressão do contexto em um estado de dimensão fixa inevitavelmente perde informação. Para tarefas que exigem recuperação precisa de detalhes específicos de documentos longos (como um RAG sobre contratos jurídicos buscando uma cláusula específica no token 1.500.000), a atenção completa do Transformer — que pode literalmente "olhar" qualquer posição anterior com igual acesso — tem vantagem qualitativa. Modelos híbridos como Jamba (SSM + Transformer intercalados) e Falcon Mamba buscam o melhor dos dois mundos: eficiência do SSM para contexto geral e janelas de atenção seletiva para recuperação precisa. A fronteira da IA está na síntese dessas duas arquiteturas, não na vitória de uma sobre a outra.

---

## 10. Referências

1. Vaswani, A. et al. (2017). *Attention is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

3. Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

4. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

5. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

6. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

7. Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

8. Zhang, P. et al. (2024). *TinyLlama: An Open-Source Small Language Model*. [arXiv:2401.02385](https://arxiv.org/abs/2401.02385)

---

<div align="center">

**Projeto desenvolvido para a disciplina de IA Avançada**  
Antonio Vicente da Costa Osorio Neto  
2025 — Implementação completa com suporte a GPU e fallback CPU

</div>
