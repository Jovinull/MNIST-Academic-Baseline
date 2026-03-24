# MNIST Academic Baseline

Pipeline acadêmico de ponta a ponta para classificação de dígitos manuscritos do dataset MNIST, construído a partir dos arquivos binários originais (formato IDX). O projeto foi concebido como referência didática de nível de pós-graduação, com código extensivamente documentado, múltiplas arquiteturas de redes neurais e rotina avançada de otimização de hiperparâmetros.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Pré-requisitos e Instalação](#pré-requisitos-e-instalação)
4. [Download dos Dados](#download-dos-dados)
5. [Guia de Execução](#guia-de-execução)
6. [Arquiteturas Implementadas](#arquiteturas-implementadas)
7. [Estratégia de Experimentação](#estratégia-de-experimentação)
8. [Otimização de Hiperparâmetros](#otimização-de-hiperparâmetros--estratégias-e-mecanismo-interno)
9. [Pipeline Detalhado](#pipeline-detalhado)
10. [Referências Bibliográficas](#referências-bibliográficas)

---

## Visão Geral

Este repositório implementa um pipeline completo de Machine Learning para classificação supervisionada dos 10 dígitos (0–9) do MNIST, seguindo as melhores práticas da academia e da engenharia de Deep Learning. O projeto foi estruturado para que cada módulo seja autocontido, extensivamente comentado e rastreável a referências da literatura.

### Características Principais

- **Parsing direto dos binários IDX**: leitura dos arquivos `.idx3-ubyte` e `.idx1-ubyte` sem depender de datasets prontos do `torchvision`, demonstrando compreensão do formato de baixo nível.
- **4 arquiteturas progressivas**: MLP → LeNet-5 → ModernCNN → DeepCNN (com blocos residuais), refletindo a evolução histórica do campo.
- **Tuning avançado com Optuna**: busca bayesiana (TPE), random search e grid search sobre 8 dimensões de hiperparâmetros, com pruning automático de trials.
- **Avaliação rigorosa**: métricas por classe (precision, recall, F1), matriz de confusão, curvas de aprendizado e análise qualitativa de erros.
- **Reprodutibilidade total**: fixação de sementes em todos os geradores aleatórios (Python, NumPy, PyTorch, cuDNN).

---

## Estrutura do Projeto

```
MNIST-Academic-Baseline/
│
├── main.py                          # Ponto de entrada do pipeline
├── requirements.txt                 # Dependências do projeto
├── README.md                        # Este arquivo
│
├── config/
│   └── hyperparameters.yaml         # Configuração centralizada de hiperparâmetros
│
├── data/
│   └── raw/                         # Arquivos binários IDX do MNIST
│       ├── train-images.idx3-ubyte  #   60.000 imagens de treino
│       ├── train-labels.idx1-ubyte  #   60.000 rótulos de treino
│       ├── t10k-images.idx3-ubyte   #   10.000 imagens de teste
│       └── t10k-labels.idx1-ubyte   #   10.000 rótulos de teste
│
├── src/
│   ├── __init__.py                  # Inicialização do pacote
│   ├── data_loader.py               # Parsing IDX + Dataset + DataLoaders
│   ├── preprocessing.py             # Normalização e data augmentation
│   ├── architectures.py             # MLP, LeNet-5, ModernCNN, DeepCNN
│   ├── training.py                  # Loop de treino/validação + Trainer
│   ├── evaluation.py                # Métricas, gráficos e relatórios
│   └── tuning.py                    # Otimização de hiperparâmetros (Optuna)
│
└── results/                         # Gerado automaticamente
    ├── figures/                     #   Gráficos (curvas, matrizes, erros)
    ├── checkpoints/                 #   Pesos dos melhores modelos (.pt)
    └── logs/                        #   Logs de experimentos
```

### Descrição dos Módulos

| Módulo | Responsabilidade | Conceitos-chave |
|--------|-----------------|-----------------|
| `data_loader.py` | Leitura binária IDX, conversão para tensores, particionamento treino/val/teste | Formato IDX, `struct.unpack`, `Dataset`, `DataLoader`, `random_split` |
| `preprocessing.py` | Normalização z-score, data augmentation (rotação, translação) | Internal Covariate Shift, regularização implícita, invariância |
| `architectures.py` | 4 arquiteturas de redes neurais com complexidade crescente | CNN, campos receptivos, BatchNorm, Dropout, conexões residuais |
| `training.py` | Loop de treinamento, otimizadores, schedulers, early stopping | SGD, Adam, AdamW, backpropagation, learning rate scheduling |
| `evaluation.py` | Métricas de classificação e visualizações científicas | Precision, Recall, F1, matriz de confusão, learning curves |
| `tuning.py` | Busca automatizada de hiperparâmetros com Optuna | TPE, Random Search, Grid Search, pruning bayesiano |

---

## Pré-requisitos e Instalação

### Requisitos de Sistema

- Python 3.8+
- pip ou conda
- (Opcional) GPU NVIDIA com CUDA para aceleração

### Instalação

```bash
# 1. Clone o repositório (ou navegue até o diretório)
cd MNIST-Academic-Baseline

# 2. (Recomendado) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Download dos Dados

Os 4 arquivos binários IDX do MNIST devem ser colocados em `data/raw/`. Eles podem ser obtidos diretamente do site oficial:

```bash
cd data/raw

# Download dos arquivos (descompacte se necessário)
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Descompactar
gunzip *.gz
```

Após a descompactação, o diretório `data/raw/` deve conter:
- `train-images.idx3-ubyte` (≈47 MB)
- `train-labels.idx1-ubyte` (≈60 KB)
- `t10k-images.idx3-ubyte` (≈7.8 MB)
- `t10k-labels.idx1-ubyte` (≈10 KB)

---

## Guia de Execução

### Modo 1: Treinamento Único

Treina um modelo com os hiperparâmetros padrão definidos em `config/hyperparameters.yaml`:

```bash
# Treinamento com configuração padrão (LeNet-5, Adam, lr=1e-3, 20 épocas)
python main.py --mode train

# Especificando arquitetura e hiperparâmetros via CLI
python main.py --mode train --architecture DeepCNN --epochs 30 --learning_rate 0.001

# Usando GPU (se disponível)
python main.py --mode train --architecture ModernCNN --device cuda

# Todas as opções de arquitetura
python main.py --mode train --architecture MLP
python main.py --mode train --architecture LeNet5
python main.py --mode train --architecture ModernCNN
python main.py --mode train --architecture DeepCNN
```

### Modo 2: Otimização de Hiperparâmetros

Executa a busca automatizada com Optuna:

```bash
# Tuning com 50 trials usando TPE (bayesiano) — recomendado
python main.py --mode tune --n_trials 50 --sampler tpe

# Tuning com Random Search (baseline de comparação)
python main.py --mode tune --n_trials 30 --sampler random

# Tuning com Grid Search (exaustivo — pode ser demorado)
python main.py --mode tune --sampler grid

# Tuning em GPU
python main.py --mode tune --n_trials 50 --sampler tpe --device cuda
```

### Argumentos Disponíveis

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--mode` | str | `train` | `train` ou `tune` |
| `--config` | str | `config/hyperparameters.yaml` | Caminho do YAML |
| `--data_dir` | str | `data/raw` | Diretório dos dados IDX |
| `--output_dir` | str | `results` | Diretório de saída |
| `--device` | str | `auto` | `cpu`, `cuda`, `mps` ou `auto` |
| `--architecture` | str | YAML | `MLP`, `LeNet5`, `ModernCNN`, `DeepCNN` |
| `--epochs` | int | YAML | Número de épocas |
| `--learning_rate` | float | YAML | Taxa de aprendizado |
| `--batch_size` | int | YAML | Tamanho do mini-batch |
| `--optimizer` | str | YAML | `SGD`, `Adam`, `AdamW` |
| `--n_trials` | int | YAML | Número de trials (modo tune) |
| `--sampler` | str | `tpe` | `tpe`, `random`, `grid` |
| `--no_saved_params` | flag | — | Ignora `best_params.yaml` e usa apenas defaults do YAML |

---

## Arquiteturas Implementadas

### 1. MLP (Multilayer Perceptron) — Baseline

Rede feedforward com camadas totalmente conectadas. Achata a imagem 28×28 em um vetor de 784 dimensões, perdendo toda informação espacial. Serve como baseline para quantificar o ganho das convoluções.

- **Parâmetros**: ~270K
- **Acurácia esperada**: ~97.5–98.0%
- **Referência**: Rumelhart, Hinton & Williams (1986)

### 2. LeNet-5 — CNN Clássica

A arquitetura convolucional pioneira para reconhecimento de dígitos, adaptada com ativações ReLU e max pooling. Introduz campos receptivos locais e compartilhamento de pesos.

- **Parâmetros**: ~44K
- **Acurácia esperada**: ~98.5–99.0%
- **Referência**: LeCun, Bottou, Bengio & Haffner (1998)

### 3. ModernCNN — CNN com BatchNorm + Dropout

CNN moderna com duas convoluções por bloco, Batch Normalization, Dropout2d espacial e canais progressivamente mais largos (32 → 64).

- **Parâmetros**: ~160K
- **Acurácia esperada**: ~99.2–99.4%
- **Referências**: Ioffe & Szegedy (2015), Srivastava et al. (2014)

### 4. DeepCNN — CNN Profunda com Blocos Residuais

Arquitetura inspirada na ResNet com conexões skip que permitem treinar redes mais profundas sem degradação. Usa 10 camadas convolucionais com Adaptive Average Pooling.

- **Parâmetros**: ~125K
- **Acurácia esperada**: ~99.3–99.5%
- **Referência**: He, Zhang, Ren & Sun (2016)

---

## Estratégia de Experimentação

### Abordagem Sistemática

A experimentação segue uma abordagem em duas fases:

**Fase 1 — Treinamento Individual (Baseline)**
1. Treinar cada arquitetura (MLP, LeNet-5, ModernCNN, DeepCNN) com os hiperparâmetros padrão do YAML.
2. Comparar acurácias de teste para quantificar o impacto de cada inovação arquitetural.
3. Analisar as curvas de aprendizado para detectar underfitting/overfitting.

**Fase 2 — Otimização de Hiperparâmetros**
1. Executar o tuning com TPE (50+ trials) sobre o espaço completo de 8 dimensões.
2. Identificar a melhor configuração global.
3. Comparar TPE vs. Random Search para validar a eficácia da busca bayesiana.

### Espaço de Busca (8 dimensões)

| Hiperparâmetro | Tipo | Intervalo / Opções | Fundamentação |
|----------------|------|-------------------|---------------|
| Arquitetura | Categórico | MLP, LeNet5, ModernCNN, DeepCNN | Progressão histórica |
| Learning Rate | Log-uniforme | [1e-4, 1e-1] | Smith (2018) |
| Batch Size | Categórico | 32, 64, 128, 256 | Keskar et al. (2017) |
| Otimizador | Categórico | SGD, Adam, AdamW | Kingma & Ba (2015) |
| Épocas | Categórico | 10, 20, 30 | Early stopping complementa |
| Dropout Rate | Uniforme | [0.0, 0.5] | Srivastava et al. (2014) |
| Weight Decay | Log-uniforme | [1e-6, 1e-2] | Krogh & Hertz (1991) |
| Scheduler | Categórico | StepLR, Cosine, Plateau, None | Loshchilov & Hutter (2017) |

### Métricas de Avaliação

- **Acurácia global**: fração de predições corretas sobre todo o conjunto de teste.
- **Precision, Recall, F1-Score por classe**: métricas detalhadas que revelam desempenho heterogêneo entre dígitos.
- **Matriz de confusão**: identifica padrões de confusão sistemáticos (e.g., 4↔9, 3↔8).
- **Análise qualitativa de erros**: visualização dos exemplos incorretamente classificados.

---

## Otimizacao de Hiperparametros — Mecanismo Interno e Estrategias

Esta secao explica em detalhe como o pipeline resolve os hiperparametros em cada modo de execucao, como os resultados de tuning sao persistidos e comparados entre execucoes, qual a relacao entre o treinamento do tuning e o treinamento normal, e como os artefatos sao nomeados.

---

### Persistencia Automatica dos Melhores Hiperparametros

O pipeline possui um sistema de persistencia implementado no modulo `src/persistence.py` que conecta o tuning ao treinamento entre execucoes independentes. O mecanismo funciona da seguinte forma:

1. Ao final de um `--mode tune`, o melhor resultado e salvo em `results/best_params.yaml` contendo os 8 hiperparametros, metricas de desempenho (acuracia, numero de parametros, tempo de treinamento) e metadados (sampler, numero de trials, timestamp).

2. Ao executar `--mode train`, o pipeline verifica automaticamente se `results/best_params.yaml` existe. Se existir, carrega os hiperparametros desse arquivo e os utiliza no lugar dos defaults do YAML. Nenhuma intervencao manual e necessaria.

3. Ao executar um novo `--mode tune`, o resultado e comparado automaticamente contra o `best_params.yaml` existente. O novo resultado so sobrescreve o anterior se for estritamente superior. Caso contrario, o anterior e mantido.

4. Todas as execucoes de tuning sao registradas em `results/tuning_history.yaml`, independente de serem melhores ou nao.

---

### Resolucao de Hiperparametros: Hierarquia de 3 Niveis

A funcao `run_single_training` em `main.py` resolve os hiperparametros seguindo esta ordem de prioridade:

```
Nivel 1 (menor prioridade):  config/hyperparameters.yaml  ->  secao "defaults"
Nivel 2:                     results/best_params.yaml      ->  gerado pelo tuning
Nivel 3 (maior prioridade):  argumentos de linha de comando (--architecture, etc.)
```

O codigo que implementa esta resolucao:

```python
# main.py, funcao run_single_training

# 1. Carrega defaults do YAML
defaults = config['defaults'].copy()

# 2. Se best_params.yaml existir, sobrescreve os defaults
saved_best = load_best_params(args.output_dir)
if saved_best is not None:
    defaults.update(saved_best['hyperparameters'])

# 3. Argumentos CLI tem prioridade maxima
architecture  = args.architecture or defaults['architecture']
epochs        = args.epochs       or defaults['epochs']
learning_rate = args.learning_rate or defaults['learning_rate']
batch_size    = args.batch_size   or defaults['batch_size']
optimizer_name = args.optimizer   or defaults['optimizer']
dropout_rate  = defaults['dropout_rate']
weight_decay  = defaults['weight_decay']
scheduler_name = defaults['scheduler']
```

O operador `or` do Python retorna o primeiro valor truthy. Se um argumento CLI foi fornecido, ele prevalece. Se nao, o valor de `defaults` (que pode ter sido atualizado pelo `best_params.yaml`) e usado.

---

### Cenarios de Uso Detalhados

**Cenario 1: Treinamento sem tuning previo**

```bash
python main.py --mode train
```

Nenhum `best_params.yaml` existe. O pipeline usa exclusivamente a secao `defaults` do YAML:

```yaml
defaults:
  architecture: "LeNet5"
  learning_rate: 1.0e-3
  batch_size: 64
  optimizer: "Adam"
  epochs: 20
  dropout_rate: 0.25
  weight_decay: 1.0e-4
  scheduler: "CosineAnnealingLR"
```

**Cenario 2: Tuning seguido de treinamento**

```bash
python main.py --mode tune --n_trials 50 --sampler tpe    # (1) Tuning
python main.py --mode train                                # (2) Treinamento
```

Execucao (1): o tuning testa 50 combinacoes, encontra a melhor (e.g., DeepCNN com lr=2.8e-3), salva os resultados em `results/best_params.yaml`, e treina automaticamente o modelo final com esses parametros.

Execucao (2): o pipeline detecta `results/best_params.yaml`, carrega os hiperparametros otimizados (DeepCNN, lr=2.8e-3, etc.), e treina com eles. O log exibe:

```
Hiperparametros otimizados detectados em results/best_params.yaml
(val_acc: 0.9945, tuning de: 2026-03-23 14:30:00). Usando como base para o treinamento.
```

Nenhuma copia manual de parametros e necessaria. Todos os 8 hiperparametros (incluindo `dropout_rate`, `weight_decay` e `scheduler`, que nao possuem flags CLI) sao carregados automaticamente.

**Cenario 3: Segundo tuning — comparacao automatica**

```bash
python main.py --mode tune --n_trials 50 --sampler tpe      # (1) Primeiro tuning
python main.py --mode tune --n_trials 50 --sampler random    # (2) Segundo tuning
```

Execucao (1): salva o melhor resultado em `best_params.yaml` (e.g., val_acc=0.9932).

Execucao (2): ao terminar, o pipeline carrega o `best_params.yaml` do tuning anterior e compara com o novo resultado. O log exibe uma tabela de comparacao:

```
======================================================================
COMPARACAO: Novo Tuning vs. Melhor Salvo Anteriormente
======================================================================

  Metrica                      Salvo           Novo       Melhor
  ──────────────────────── ────────────── ────────────── ──────────
  Val Accuracy                   0.9932         0.9945       NOVO
  Num Parameters                128,456         95,200       NOVO
  Training Time (s)              142.5          138.2        NOVO
  ──────────────────────── ────────────── ────────────── ──────────
  Architecture                ModernCNN        DeepCNN        ---
  Sampler                          tpe         random         ---
  Trials                            50             50         ---
  Timestamp              2026-03-23 14:30 2026-03-23 16:00    ---

======================================================================
  VEREDITO: Novo resultado e SUPERIOR ao salvo anteriormente.
  Atualizando best_params.yaml.
======================================================================
```

Se o novo resultado for inferior, o `best_params.yaml` anterior e mantido intacto:

```
  VEREDITO: Resultado salvo anteriormente e IGUAL ou SUPERIOR.
  best_params.yaml mantido sem alteracao.
```

**Cenario 4: Treinamento ignorando o tuning**

```bash
python main.py --mode train --no_saved_params
```

A flag `--no_saved_params` faz com que o pipeline ignore o `best_params.yaml` e use exclusivamente os defaults do YAML. O log exibe:

```
Flag --no_saved_params ativa. Ignorando best_params.yaml e usando defaults do YAML.
```

Isso permite comparar o desempenho do modelo com e sem otimizacao de hiperparametros.

**Cenario 5: Treinamento com override parcial via CLI**

```bash
python main.py --mode train --architecture MLP --epochs 30
```

Se `best_params.yaml` existir (e.g., com architecture=DeepCNN, epochs=20), os argumentos CLI tem prioridade: architecture=MLP (da CLI), epochs=30 (da CLI), mas learning_rate, dropout_rate, weight_decay, scheduler, batch_size e optimizer vem do `best_params.yaml`.

---

### Criterios de Comparacao entre Tunings

A funcao `_is_strictly_better` em `src/persistence.py` usa uma hierarquia de tres criterios para decidir se um novo resultado substitui o anterior:

1. **Acuracia de validacao** (criterio primario): o resultado com maior `val_accuracy` vence. Diferencas menores que 0.01% (ACCURACY_THRESHOLD=0.0001) sao tratadas como empate para evitar que ruido de ponto flutuante cause trocas desnecessarias.

2. **Numero de parametros** (desempate - Navalha de Occam): se as acuracias forem equivalentes, o modelo com menos parametros treinaveis e preferido. Modelos mais simples tendem a generalizar melhor (Goodfellow et al., 2016, cap. 5.6).

3. **Tempo de treinamento** (desempate secundario): se acuracia e complexidade forem equivalentes, o modelo mais rapido de treinar e preferido.

```python
# src/persistence.py, funcao _is_strictly_better
if new_acc > old_acc + 0.0001:    return True   # Novo tem acuracia superior
if old_acc > new_acc + 0.0001:    return False   # Antigo tem acuracia superior
if new_params < old_params:       return True   # Empate em acc: prefere menos parametros
if old_params < new_params:       return False
return new_time < old_time                       # Empate total: prefere mais rapido
```

---

### Historico Cumulativo de Tunings

Cada execucao de tuning e registrada em `results/tuning_history.yaml`, independente de ter se tornado o novo melhor ou nao. Cada entrada contem todos os hiperparametros, metricas, metadados e um campo `was_saved_as_best` indicando se aquela execucao substituiu o melhor anterior.

Este arquivo permite analise retrospectiva: qual sampler tende a produzir melhores resultados? Quantos trials sao suficientes? Qual arquitetura aparece com mais frequencia entre os melhores?

---

### Conteudo do Arquivo `best_params.yaml`

```yaml
hyperparameters:
  architecture: DeepCNN
  learning_rate: 0.00234
  batch_size: 128
  optimizer: AdamW
  epochs: 30
  dropout_rate: 0.18
  weight_decay: 3.2e-05
  scheduler: CosineAnnealingLR

metrics:
  val_accuracy: 0.9945
  num_parameters: 128456
  training_time_seconds: 142.5

metadata:
  sampler: tpe
  n_trials: 50
  n_completed: 47
  n_pruned: 3
  trial_number: 38
  timestamp: '2026-03-23 14:30:00'
```

---

### Relacao entre o Treinamento do Tuning e o Treinamento Normal

O treinamento executado dentro de cada trial do tuning e o treinamento do modo `--mode train` utilizam **exatamente o mesmo codigo**. Ambos invocam:

- `build_model` (de `architectures.py`) para instanciar a rede neural.
- `build_optimizer` e `build_scheduler` (de `training.py`) para configurar o otimizador e o scheduler.
- A classe `Trainer` (de `training.py`) com o mesmo loop de epocas, early stopping e checkpoint.
- As funcoes `train_one_epoch` e `validate` (de `training.py`) para cada epoca.

A unica diferenca e a **origem dos hiperparametros**:

| Aspecto | `--mode train` | Trial do tuning | Treinamento final do tuning |
|---------|---------------|-----------------|---------------------------|
| Origem dos hiperparametros | best_params.yaml > YAML defaults > CLI | Optuna `trial.suggest_*` | `study.best_trial.params` injetados em `args`/`config` |
| Funcao de treinamento | `run_single_training` | `Trainer.fit` (dentro de `objective`) | `run_single_training` |
| Avaliacao no teste | Sim (via `full_evaluation`) | Nao (apenas validacao) | Sim (via `full_evaluation`) |
| Gera figuras e relatorio | Sim | Nao | Sim |
| Salva checkpoint | Sim | Nao | Sim |

Os trials individuais do tuning NAO avaliam no conjunto de teste e NAO geram figuras. Apenas a acuracia de validacao e reportada ao Optuna. O treinamento final (Fase 2 do tune) e que executa a avaliacao completa.

---

### Nomenclatura dos Artefatos de Saida

Os arquivos de saida seguem o padrao `{nome_da_arquitetura}_*`. O nome e determinado pela variavel `architecture` resolvida durante o treinamento:

| Cenario | Arquivo gerado |
|---------|---------------|
| `--mode train` (default YAML: LeNet5) | `results/checkpoints/LeNet5_best.pt` |
| `--mode train` (carregou best_params: DeepCNN) | `results/checkpoints/DeepCNN_best.pt` |
| `--mode train --architecture MLP` | `results/checkpoints/MLP_best.pt` |
| `--mode tune` (melhor trial: ModernCNN) | `results/checkpoints/ModernCNN_best.pt` |

Nao ha distincao de nome entre um modelo treinado via `--mode train` e o treinamento final do `--mode tune`. Se ambos usarem a mesma arquitetura, o segundo sobrescreve o primeiro.

Artefatos completos gerados em `--output_dir`:

```
results/
  best_params.yaml                       # Melhores hiperparametros (persistente entre execucoes)
  tuning_history.yaml                    # Historico cumulativo de todos os tunings
  tuning_report.txt                      # Relatorio detalhado do tuning mais recente
  {arch}_report.txt                      # Relatorio de metricas do treinamento
  figures/
    {arch}_learning_curves.png           # Curvas de perda e acuracia
    {arch}_confusion_matrix.png          # Matriz de confusao
    {arch}_misclassified.png             # Exemplos incorretamente classificados
  checkpoints/
    {arch}_best.pt                       # Pesos do modelo (state_dict)
  logs/
    experiment.log                       # Log completo (modo append)
```

---

### Carregamento de um Modelo Salvo

O checkpoint salvo em `results/checkpoints/{arch}_best.pt` contem um dicionario com as seguintes chaves:

```python
{
    'epoch':           25,                    # Epoca em que o melhor resultado foi atingido
    'model_state':     model.state_dict(),    # Pesos da rede neural
    'optimizer_state':  optimizer.state_dict(), # Estado interno do otimizador
    'val_loss':        0.032,                 # Perda de validacao da melhor epoca
    'val_acc':         0.9940,                # Acuracia de validacao da melhor epoca
}
```

Para carregar o modelo em codigo Python:

```python
import torch
from src.architectures import build_model

# A arquitetura e o dropout_rate devem corresponder ao modelo original.
# Essas informacoes estao no tuning_report.txt ou no log de treinamento.
model = build_model('DeepCNN', dropout_rate=0.22)
checkpoint = torch.load('results/checkpoints/DeepCNN_best.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

---

### Comparacao dos Tres Samplers

O pipeline oferece tres estrategias de exploracao do espaco de hiperparametros, cada uma com caracteristicas distintas de cobertura, eficiencia e garantias teoricas.

#### TPE (Tree-structured Parzen Estimator)

Estrategia de otimizacao bayesiana baseada em modelos de densidade condicional. Nos primeiros N trials (por padrao, 10), o TPE amostra aleatoriamente para construir um modelo inicial. A partir dai, ele divide os trials observados em dois grupos -- os de melhor desempenho (top 25%) e os restantes -- e ajusta duas distribuicoes de probabilidade: `l(x)` para os bons e `g(x)` para os ruins. Novos hiperparametros sao amostrados maximizando a razao `l(x)/g(x)`, concentrando a busca nas regioes mais promissoras.

- **Cobertura do espaco**: parcial e adaptativa. Nao testa todas as combinacoes; foca progressivamente nas regioes de alto desempenho.
- **Vantagem**: converge mais rapidamente que os demais, pois aprende com os resultados anteriores.
- **Desvantagem**: pode convergir para otimos locais se o espaco de busca for altamente multimodal.
- **Referencia**: Bergstra, Bardenet, Bengio & Kegl (2011), "Algorithms for Hyper-Parameter Optimization", NeurIPS.

#### Random Search

Amostra cada hiperparametro independentemente a partir de sua distribuicao definida no espaco de busca (uniforme, log-uniforme ou categorica). Cada trial e independente dos anteriores -- nao ha aprendizado entre trials.

- **Cobertura do espaco**: estocastica. A probabilidade de cobrir uma regiao e proporcional ao numero de trials, mas nao ha garantia de cobertura uniforme.
- **Vantagem**: demonstrado por Bergstra & Bengio (2012) como superior ao Grid Search na maioria dos cenarios, pois cada trial explora uma combinacao inedita de todos os hiperparametros, enquanto o Grid repete valores nos eixos irrelevantes.
- **Desvantagem**: desperdicio computacional em regioes do espaco com desempenho consistentemente ruim.
- **Referencia**: Bergstra & Bengio (2012), "Random Search for Hyper-Parameter Optimization", JMLR 13.

#### Grid Search

Define um grid discreto explicito de valores para cada hiperparametro e avalia todas as combinacoes possiveis (produto cartesiano). Ao selecionar `--sampler grid`, o codigo em `tuning.py` constroi o seguinte grid fixo:

```python
grid = {
    'architecture': ['MLP', 'LeNet5', 'ModernCNN', 'DeepCNN'],  # 4 valores
    'learning_rate': [1e-4, 1e-3, 1e-2],                        # 3 valores
    'batch_size': [32, 64, 128, 256],                            # 4 valores
    'optimizer': ['SGD', 'Adam', 'AdamW'],                       # 3 valores
    'epochs': [10, 20, 30],                                      # 3 valores
    'dropout_rate': [0.0, 0.25, 0.5],                            # 3 valores
    'weight_decay': [1e-5, 1e-4, 1e-3],                         # 3 valores
    'scheduler': ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'None'],  # 4 valores
}
# Total: 4 x 3 x 4 x 3 x 3 x 3 x 3 x 4 = 15.552 combinacoes
```

- **Cobertura do espaco**: completa dentro do grid definido. Todas as combinacoes sao avaliadas.
- **Vantagem**: garante que a melhor combinacao do grid sera encontrada.
- **Desvantagem**: custo computacional exponencial. Hiperparametros continuos (learning rate, dropout) precisam ser discretizados, perdendo granularidade.
- **Referencia**: Bergstra & Bengio (2012) demonstram formalmente que Grid Search e estatisticamente inferior a Random Search.

### Tabela Comparativa dos Samplers

| Aspecto | TPE | Random Search | Grid Search |
|---------|-----|---------------|-------------|
| Mecanismo | Otimizacao bayesiana (modelo de densidade) | Amostragem aleatoria independente | Produto cartesiano de um grid discreto |
| Cobertura | Parcial, adaptativa | Parcial, estocastica | Completa (dentro do grid) |
| Aprende com trials anteriores | Sim | Nao | Nao |
| Suporta parametros continuos | Sim (amostragem direta) | Sim (amostragem direta) | Nao (requer discretizacao) |
| Custo para 50 trials | ~1.5h | ~1.5h | N/A (grid define o total) |
| Custo para cobertura completa | Nao se aplica | Nao se aplica | Proporcional ao produto cartesiano |
| Garantia de otimalidade | Nao (otimo local) | Nao (probabilistico) | Sim, dentro do grid definido |
| Caso de uso principal | Exploracao eficiente de espacos grandes | Baseline experimental | Espacos pequenos ou analise exaustiva |
| Referencia | Bergstra et al. (2011) | Bergstra & Bengio (2012) | -- |

---

## Pipeline Detalhado

```
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE END-TO-END                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ENTRADA         Arquivos binários IDX (raw bytes)       │
│       ↓             struct.unpack → NumPy arrays            │
│                                                             │
│  2. PRÉ-PROCESSAMENTO  uint8 [0,255] → float32 [0,1]       │
│       ↓                 z-score normalization                │
│       ↓                 data augmentation (treino only)      │
│                                                             │
│  3. PARTICIONAMENTO    60K → 54K treino + 6K validação      │
│       ↓                10K teste (intocado)                  │
│                                                             │
│  4. MODELO             MLP / LeNet-5 / ModernCNN / DeepCNN  │
│       ↓                                                     │
│  5. TREINAMENTO        forward → loss → backward → update   │
│       ↓                early stopping + checkpoint           │
│       ↓                learning rate scheduling              │
│                                                             │
│  6. AVALIAÇÃO          Métricas + Gráficos + Relatório      │
│       ↓                                                     │
│  7. TUNING (opcional)  Optuna TPE / Random / Grid            │
│                        Pruning de trials pouco promissores   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Referências Bibliográficas

### Papers Fundamentais (Arquiteturas e Técnicas)

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P.** (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278–2324.
   — *Origem da LeNet-5 e do dataset MNIST. Fundamento de toda a arquitetura convolucional do projeto.*

2. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. arXiv:1512.03385.
   — *Blocos residuais (skip connections) usados na DeepCNN. Best Paper Award CVPR 2016.*

3. **Ioffe, S. & Szegedy, C.** (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.
   — *Batch Normalization usado em ModernCNN e DeepCNN.*

4. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.** (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research*, 15, 1929–1958.
   — *Técnica de regularização Dropout presente em todas as arquiteturas.*

5. **Rumelhart, D.E., Hinton, G.E., & Williams, R.J.** (1986). "Learning Representations by Back-propagating Errors." *Nature*, 323, 533–536.
   — *Algoritmo de backpropagation, base do treinamento de todas as redes neurais do projeto.*

6. **Simonyan, K. & Zisserman, A.** (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *International Conference on Learning Representations (ICLR)*. arXiv:1409.1556.
   — *Princípio de empilhar convoluções 3×3 e duplicar canais, usado na ModernCNN.*

7. **Lin, M., Chen, Q., & Yan, S.** (2014). "Network In Network." *International Conference on Learning Representations (ICLR)*. arXiv:1312.4400.
   — *Convoluções 1×1 (pointwise) e Global Average Pooling, usados na DeepCNN.*

### Papers de Otimização e Treinamento

8. **Kingma, D.P. & Ba, J.** (2015). "Adam: A Method for Stochastic Optimization." *International Conference on Learning Representations (ICLR)*. arXiv:1412.6980.
   — *Otimizador Adam, default do pipeline.*

9. **Loshchilov, I. & Hutter, F.** (2019). "Decoupled Weight Decay Regularization." *International Conference on Learning Representations (ICLR)*. arXiv:1711.05101.
   — *Otimizador AdamW com weight decay desacoplado.*

10. **Loshchilov, I. & Hutter, F.** (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *International Conference on Learning Representations (ICLR)*. arXiv:1608.03983.
    — *Cosine Annealing scheduler usado no pipeline.*

11. **Sutskever, I., Martens, J., Dahl, G., & Hinton, G.** (2013). "On the Importance of Initialization and Momentum in Deep Learning." *Proceedings of the 30th International Conference on Machine Learning (ICML)*.
    — *Momentum e Nesterov momentum no SGD.*

12. **Smith, L.N.** (2018). "A Disciplined Approach to Neural Network Hyper-Parameters: Part 1 — Learning Rate, Batch Size, Momentum, and Weight Decay." arXiv:1803.09820.
    — *Heurísticas para seleção dos intervalos do espaço de busca.*

13. **Bengio, Y.** (2012). "Practical Recommendations for Gradient-Based Training of Deep Architectures." *Neural Networks: Tricks of the Trade*, Springer, 437–478.
    — *Recomendações práticas que guiaram os defaults do YAML.*

### Papers de Otimização de Hiperparâmetros

14. **Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M.** (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
    — *Framework Optuna usado para a busca de hiperparâmetros.*

15. **Bergstra, J. & Bengio, Y.** (2012). "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13, 281–305.
    — *Fundamentação teórica de Random Search vs. Grid Search.*

16. **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B.** (2011). "Algorithms for Hyper-Parameter Optimization." *Advances in Neural Information Processing Systems (NeurIPS)*.
    — *TPE (Tree-structured Parzen Estimator), sampler bayesiano do Optuna.*

### Livros-Texto de Referência

17. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Disponível em: https://www.deeplearningbook.org/
    — *Referência principal para fundamentos teóricos. Capítulos 5 (ML Basics), 6 (Feedforward Networks), 7 (Regularização), 8 (Otimização) e 9 (CNNs) foram diretamente consultados.*

18. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer.
    — *Capítulo 7 sobre validação cruzada e seleção de modelos.*

19. **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
    — *Fundamentação probabilística de classificação e regularização.*

### Papers Complementares

20. **Glorot, X., Bordes, A., & Bengio, Y.** (2011). "Deep Sparse Rectifier Neural Networks." *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
    — *Análise da ativação ReLU usada em todas as arquiteturas.*

21. **Simard, P.Y., Steinkraus, D., & Platt, J.C.** (2003). "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis." *Proceedings of the 7th International Conference on Document Analysis and Recognition (ICDAR)*.
    — *Estratégias de data augmentation para dígitos manuscritos.*

22. **Keskar, N.S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P.T.P.** (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *International Conference on Learning Representations (ICLR)*.
    — *Impacto do tamanho do batch na generalização.*

23. **Prechelt, L.** (1998). "Early Stopping — But When?" *Neural Networks: Tricks of the Trade*, Springer, 55–69.
    — *Critérios de parada antecipada implementados no Trainer.*

24. **Paszke, A., Gross, S., Massa, F., et al.** (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems (NeurIPS)*.
    — *Framework utilizado para toda a implementação.*

25. **Bouthillier, X., Delaunay, P., Bronzi, M., et al.** (2021). "Accounting for Variance in Machine Learning Benchmarks." *Proceedings of Machine Learning and Systems (MLSys)*.
    — *Importância da reprodutibilidade e fixação de sementes aleatórias.*

26. **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830.
    — *Biblioteca usada para cálculo de métricas de classificação.*

27. **Harris, C.R., Millman, K.J., van der Walt, S.J., et al.** (2020). "Array programming with NumPy." *Nature*, 585, 357–362.
    — *Biblioteca base para manipulação de arrays no parsing IDX.*

28. **Shorten, C. & Khoshgoftaar, T.M.** (2019). "A survey on Image Data Augmentation for Deep Learning." *Journal of Big Data*, 6, 60.
    — *Survey sobre técnicas de data augmentation.*

---

## Licença

Este projeto é de uso acadêmico e educacional.

---

*Projeto desenvolvido como baseline acadêmico para classificação MNIST, com foco em legibilidade, reprodutibilidade e rigor metodológico.*
