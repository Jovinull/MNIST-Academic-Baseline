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

Esta secao explica em detalhe como o pipeline resolve os hiperparametros em cada modo de execucao, qual a relacao entre o tuning e o treinamento, como os artefatos sao nomeados e salvos, e o que acontece quando multiplos tunings sao executados.

---

### Origem dos Hiperparametros no Modo `--mode train`

O comando `--mode train` utiliza **exclusivamente parametros fixos**. Nao consulta nenhum resultado de tuning anterior, nao le nenhum arquivo de relatorio, e nao possui nenhum mecanismo de persistencia entre execucoes. Os valores sao resolvidos pela funcao `run_single_training` em `main.py` (linhas 347-357) seguindo esta logica:

```python
defaults = config['defaults']                                  # Le a secao "defaults" do YAML
architecture  = args.architecture or defaults['architecture']  # CLI tem prioridade sobre YAML
epochs        = args.epochs       or defaults['epochs']
learning_rate = args.learning_rate or defaults['learning_rate']
batch_size    = args.batch_size   or defaults['batch_size']
optimizer_name = args.optimizer   or defaults['optimizer']
dropout_rate  = defaults['dropout_rate']                       # Somente via YAML (sem flag CLI)
weight_decay  = defaults['weight_decay']                       # Somente via YAML (sem flag CLI)
scheduler_name = defaults['scheduler']                         # Somente via YAML (sem flag CLI)
```

A prioridade e:

1. Se um argumento foi passado via linha de comando (e.g., `--architecture DeepCNN`), esse valor e usado.
2. Se nao foi passado, o valor da secao `defaults` do arquivo `config/hyperparameters.yaml` e usado.

Os defaults do YAML sao:

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

Esses valores foram escolhidos com base na literatura (Bengio, 2012; Smith, 2018) e representam uma configuracao conservadora que produz resultados solidos sem otimizacao.

**Consequencia direta**: executar `python main.py --mode train` apos um tuning anterior NAO utiliza os resultados daquele tuning. O modo train e completamente independente. Para que ele use parametros diferentes, e necessario passa-los explicitamente via CLI ou editar a secao `defaults` no YAML.

---

### Origem dos Hiperparametros no Modo `--mode tune`

O modo `--mode tune` executa duas fases sequenciais **dentro de uma unica execucao do script**:

**Fase 1 — Busca (funcao `run_tuning` em `tuning.py`)**

O Optuna cria um objeto `Study` em memoria e executa N trials. Em cada trial, o Optuna sugere uma combinacao de 8 hiperparametros a partir do espaco de busca definido na secao `search_space` do YAML. O codigo em `tuning.py` (funcao `objective`) chama `trial.suggest_categorical`, `trial.suggest_float`, etc., para obter cada valor. Com esses valores, o trial constroi um modelo, treina por ate N epocas (com early stopping), e retorna a acuracia de validacao ao Optuna.

Cada trial executa o loop de treinamento completo: as mesmas funcoes `train_one_epoch` e `validate` de `training.py` sao chamadas. O treinamento dentro de cada trial usa exatamente a mesma logica do modo `--mode train` — a classe `Trainer`, o forward/backward pass, o early stopping e o scheduler sao identicos. A diferenca e que os hiperparametros nao vem do YAML nem da CLI, mas do Optuna.

**Fase 2 — Treinamento final automatico (em `main.py`, linhas 509-532)**

Apos o termino de todos os trials, o script extrai os hiperparametros do melhor trial e os injeta em memoria nos objetos `args` e `config`. O codigo que faz isso e:

```python
# main.py, linhas 512-532
best_params = study.best_trial.params

# Sobrescreve os campos do objeto args (simula como se fossem argumentos CLI)
args.architecture  = best_params.get('architecture', 'LeNet5')
args.epochs        = best_params.get('epochs', 20)
args.learning_rate = best_params.get('learning_rate', 1e-3)
args.batch_size    = best_params.get('batch_size', 64)
args.optimizer     = best_params.get('optimizer', 'Adam')

# Sobrescreve a secao defaults do config em memoria (para dropout, weight_decay, scheduler)
config['defaults'].update({
    'dropout_rate':  best_params.get('dropout_rate', 0.25),
    'weight_decay':  best_params.get('weight_decay', 1e-4),
    'scheduler':     best_params.get('scheduler', 'CosineAnnealingLR'),
})

# Chama a mesma funcao de treinamento usada pelo --mode train
run_single_training(args, config, device)
```

A funcao `run_single_training` e chamada com os objetos `args` e `config` ja modificados. Quando ela executa `args.architecture or defaults['architecture']`, encontra o valor do melhor trial em `args.architecture` e o utiliza. Para `dropout_rate` (que nao tem flag CLI), encontra o valor atualizado em `config['defaults']['dropout_rate']`.

Essa injecao acontece **somente em memoria**. O arquivo `config/hyperparameters.yaml` nao e alterado. Quando o processo Python termina, esses valores sao descartados. Nao ha nenhum mecanismo que persista os melhores hiperparametros para uso futuro pelo modo `--mode train`.

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
| Origem dos hiperparametros | YAML defaults + CLI | Optuna `trial.suggest_*` | `study.best_trial.params` injetados em `args`/`config` |
| Funcao de treinamento | `run_single_training` | `Trainer.fit` (dentro de `objective`) | `run_single_training` |
| Avaliacao no teste | Sim (via `full_evaluation`) | Nao (apenas validacao) | Sim (via `full_evaluation`) |
| Gera figuras e relatorio | Sim | Nao | Sim |
| Salva checkpoint | Sim | Nao | Sim |

Nota: os trials individuais do tuning NAO avaliam no conjunto de teste e NAO geram figuras. Apenas a acuracia de validacao e reportada ao Optuna. O treinamento final (Fase 2) e que executa a avaliacao completa com metricas, figuras e checkpoint.

---

### Nomenclatura dos Artefatos de Saida

Os arquivos de saida seguem o padrao `{nome_da_arquitetura}_*`. O nome e determinado pela variavel `architecture` resolvida durante o treinamento. Exemplos:

| Cenario | Arquivo gerado |
|---------|---------------|
| `--mode train` (default YAML: LeNet5) | `results/checkpoints/LeNet5_best.pt` |
| `--mode train --architecture DeepCNN` | `results/checkpoints/DeepCNN_best.pt` |
| `--mode tune` (melhor trial: ModernCNN) | `results/checkpoints/ModernCNN_best.pt` |
| `--mode tune` (melhor trial: DeepCNN) | `results/checkpoints/DeepCNN_best.pt` |

Nao ha distincao de nome entre um modelo treinado via `--mode train` e um modelo treinado pela fase final do `--mode tune`. Se ambos usarem a mesma arquitetura (e.g., DeepCNN), o segundo a ser executado **sobrescreve** o checkpoint do primeiro no diretorio `results/checkpoints/`.

O mesmo se aplica aos demais artefatos:

```
results/
  {arch}_report.txt                      # Relatorio de metricas
  tuning_report.txt                      # Gerado apenas pelo modo tune
  figures/
    {arch}_learning_curves.png           # Curvas de perda e acuracia
    {arch}_confusion_matrix.png          # Matriz de confusao
    {arch}_misclassified.png             # Exemplos incorretamente classificados
  checkpoints/
    {arch}_best.pt                       # Pesos do modelo (state_dict)
  logs/
    experiment.log                       # Log em modo append (nao sobrescreve)
```

Para evitar sobrescrita, utilize `--output_dir` com caminhos distintos para cada execucao.

---

### Comportamento com Multiplos Tunings

Cada execucao de `--mode tune` e **completamente independente**. O Optuna cria um novo objeto `Study` em memoria a cada chamada, sem consultar resultados de execucoes anteriores. Nao ha banco de dados persistente, cache nem estado compartilhado.

**Cenario: dois tunings consecutivos com samplers diferentes**

```bash
python main.py --mode tune --sampler tpe    --n_trials 50   # Execucao A
python main.py --mode tune --sampler random --n_trials 50   # Execucao B
```

- A execucao A cria seu proprio Study, executa 50 trials, encontra o melhor, treina o modelo final, e salva os artefatos em `results/`.
- A execucao B faz o mesmo, do zero. Ela nao tem conhecimento dos resultados de A. Se o melhor trial de B encontrar a mesma arquitetura que A, os arquivos em `results/` serao sobrescritos.
- O unico arquivo que indica qual sampler foi usado e `results/tuning_report.txt`. Se B sobrescrever A, o relatorio de A e perdido.

Para preservar os resultados de ambos:

```bash
python main.py --mode tune --sampler tpe    --n_trials 50 --output_dir results/tpe
python main.py --mode tune --sampler random --n_trials 50 --output_dir results/random
```

Isso gera diretorios independentes. A comparacao entre os resultados de cada sampler e feita manualmente consultando `results/tpe/tuning_report.txt` e `results/random/tuning_report.txt`.

**Cenario: tuning seguido de `--mode train`**

```bash
python main.py --mode tune --sampler tpe --n_trials 50    # Execucao A: encontra DeepCNN com lr=2.8e-3
python main.py --mode train                                # Execucao B: usa LeNet5 com lr=1e-3 (YAML)
```

A execucao B **ignora** completamente o resultado de A. Ela le o YAML original (que nao foi modificado por A), encontra `architecture: "LeNet5"` e `learning_rate: 1.0e-3`, e treina com esses valores. O resultado do tuning so existiu em memoria durante a execucao A.

Para reproduzir manualmente o resultado de um tuning anterior, e necessario ler `tuning_report.txt` e passar os valores via CLI:

```bash
# Supondo que tuning_report.txt indica: architecture=DeepCNN, lr=0.0028, bs=256, opt=AdamW, epochs=30
python main.py --mode train \
  --architecture DeepCNN \
  --learning_rate 0.0028 \
  --batch_size 256 \
  --optimizer AdamW \
  --epochs 30
```

Os hiperparametros `dropout_rate`, `weight_decay` e `scheduler` nao possuem flags CLI. Para altera-los, edite manualmente a secao `defaults` em `config/hyperparameters.yaml` antes de executar o comando.

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
