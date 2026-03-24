# Explicacao Completa da Solucao — MNIST Academic Baseline

Este documento detalha todas as decisoes tecnicas, conceitos teoricos e justificativas de cada componente do projeto, com definicao de cada termo tecnico utilizado. Serve como referencia para defesa oral da solucao.

---

## Glossario Rapido de Termos Essenciais

| Termo | Definicao resumida |
|-------|-------------------|
| **Tensor** | Generalizacao de vetores e matrizes para N dimensoes. Um escalar e um tensor 0D, um vetor e 1D, uma matriz e 2D, uma imagem colorida e 3D (H, W, C). |
| **Batch / Mini-batch** | Subconjunto de amostras do dataset processado de uma vez. Em vez de atualizar os pesos com um exemplo por vez (SGD puro) ou com todos de uma vez (Batch GD), usa-se grupos intermediarios (ex: 64 imagens). |
| **Epoch (Epoca)** | Uma passagem completa por todo o conjunto de treino. Se o treino tem 54.000 exemplos e o batch_size e 64, uma epoca tem ceil(54000/64) = 844 iteracoes (forward + backward + update). |
| **Forward Pass** | Propagacao dos dados da entrada ate a saida atraves de todas as camadas. Produz a predicao (logits). |
| **Backward Pass** | Propagacao dos gradientes da saida ate a entrada, camada por camada, usando a regra da cadeia. Calcula ∂L/∂θ para cada parametro. |
| **Logits** | Saida bruta da ultima camada da rede, antes de qualquer normalizacao. Sao valores reais em (-∞, +∞), um por classe. |
| **Gradiente** | Vetor de derivadas parciais da funcao de perda em relacao a cada parametro. Indica a direcao e magnitude em que os pesos devem ser ajustados. |
| **Parametros / Pesos** | Variaveis treinaveis do modelo (matrizes W e vetores de bias b em camadas lineares, filtros em convolucoess). Sao ajustados pelo otimizador a cada passo. |
| **Hiperparametros** | Configuracoes definidas pelo humano antes do treino, nao aprendidas pelos dados: learning rate, batch size, numero de camadas, dropout rate, etc. |
| **Learning Rate (lr)** | Taxa de aprendizado. Controla o tamanho do passo na atualizacao dos pesos: θ ← θ - lr × ∇L. Muito alto: diverge. Muito baixo: convergencia lenta. |
| **Overfitting** | O modelo memoriza o conjunto de treino mas nao generaliza para dados novos. Sintoma: acuracia de treino alta, validacao baixa. |
| **Underfitting** | O modelo e incapaz de capturar os padroes dos dados. Sintoma: acuracia baixa em treino e validacao. |
| **Regularizacao** | Tecnicas que restringem a complexidade do modelo para melhorar generalizacao. |
| **Checkpoint** | Snapshot dos pesos do modelo em um determinado momento, salvo em disco para recuperacao posterior. |

---

## 1. Visao Geral do Pipeline

O projeto implementa um **pipeline** (sequencia encadeada de etapas de processamento) completo de **classificacao supervisionada** — onde cada imagem tem um rotulo correto conhecido — dos 10 digitos manuscritos do MNIST (0-9).

O fluxo completo e:

```
Arquivos binarios IDX (disco)
    → Parsing (leitura dos bytes brutos)
    → Arrays NumPy (N, 28, 28)
    → MNISTDataset (wrapper PyTorch)
    → DataLoader (batching + shuffling)
    → Modelo (forward pass → logits)
    → CrossEntropyLoss (calculo da perda)
    → Backpropagation (gradientes)
    → Otimizador (atualizacao dos pesos)
    → Early Stopping (monitoramento)
    → Avaliacao Final (conjunto de teste)
    → Artefatos (graficos, metricas, checkpoints)
```

O pipeline opera em dois modos:
- **`--mode train`**: treinamento unico com hiperparametros resolvidos em 3 niveis de prioridade.
- **`--mode tune`**: busca automatizada de hiperparametros com Optuna (N trials), seguida de treinamento final com a melhor configuracao encontrada.

---

## 2. Carregamento de Dados (data_loader.py)

### 2.1. Formato IDX — estrutura binaria

**IDX** e um formato binario de armazenamento de tensores multidimensionais criado por LeCun et al. (1998). Um arquivo binario e uma sequencia de bytes sem formatacao de texto — nenhum separador, nenhuma cabecalho legivel por humanos.

O MNIST e distribuido em 4 arquivos:

| Arquivo | Conteudo | Tamanho |
|---------|----------|---------|
| `train-images.idx3-ubyte` | 60.000 imagens 28x28 | ~47 MB |
| `train-labels.idx1-ubyte` | 60.000 rotulos | ~60 KB |
| `t10k-images.idx3-ubyte` | 10.000 imagens 28x28 | ~7.8 MB |
| `t10k-labels.idx1-ubyte` | 10.000 rotulos | ~10 KB |

**Estrutura do arquivo de imagens (idx3-ubyte):**

```
Offset  Tipo    Valor    Descricao
0       int32   2051     Magic number (identifica o tipo: 0x0803)
4       int32   N        Numero de imagens (60000 ou 10000)
8       int32   28       Numero de linhas (pixels)
12      int32   28       Numero de colunas (pixels)
16+     uint8   [0-255]  Pixels, N × 28 × 28 bytes contiguos
```

- **int32**: inteiro de 32 bits (4 bytes).
- **big-endian**: o byte mais significativo vem primeiro. O formato IDX usa big-endian, enquanto CPUs modernas sao little-endian — por isso e necessario especificar `>` no `struct.unpack`.
- **uint8**: unsigned int de 8 bits, valores 0-255. Cada pixel ocupa exatamente 1 byte.
- **contiguos**: os bytes estao em sequencia linear na memoria sem espacamento, o que permite leitura eficiente com `np.frombuffer`.

**Estrutura do arquivo de rotulos (idx1-ubyte):**

```
Offset  Tipo    Valor    Descricao
0       int32   2049     Magic number (0x0801)
4       int32   N        Numero de rotulos
8+      uint8   [0-9]    Rotulos, N bytes contiguos
```

### 2.2. Parsing — como a leitura funciona

```python
magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
```

- `struct.unpack`: funcao que desempacota bytes brutos em tipos Python.
- `'>IIII'`: formato de desempacotamento:
  - `>`: big-endian byte order.
  - `I`: unsigned int de 32 bits (4 bytes).
  - `IIII`: 4 ints consecutivos = 16 bytes totais.
- O resultado e uma tupla `(magic, num_images, num_rows, num_cols)`.

```python
images = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_images, 28, 28)
```

- `np.frombuffer`: interpreta um buffer de bytes diretamente como array NumPy, sem copiar os dados. Zero-copy e mais eficiente que iterar byte a byte.
- `dtype=np.uint8`: tipo de dado de cada elemento (unsigned 8-bit integer, [0,255]).
- `.reshape(N, 28, 28)`: transforma o array 1D de `N*784` bytes em um tensor 3D de forma `(N, 28, 28)`.

**Por que validar o magic number?** Para detectar arquivos corrompidos ou errados antes de tentar processar os dados, produzindo mensagens de erro claras em vez de falhas silenciosas.

### 2.3. Dataset customizado — MNISTDataset

O PyTorch usa o padrao **Dataset + DataLoader**:
- **Dataset**: sabe COMO obter uma amostra pelo indice.
- **DataLoader**: sabe QUANTAS amostras pegar de uma vez (batching) e em que ordem (shuffling).

A classe `MNISTDataset` herda de `torch.utils.data.Dataset` e implementa o contrato minimo:

```python
def __len__(self) -> int:
    # DataLoader usa isso para calcular: num_batches = ceil(len / batch_size)
    return len(self.labels)

def __getitem__(self, idx: int):
    # DataLoader chama isso para cada indice no batch atual
    image = self.images[idx]
    label = self.labels[idx]
    if self.transform is not None:
        image = self.transform(image)
    return image, label
```

**Conversoes na inicializacao:**

```python
self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
```

- `dtype=torch.float32`: converte de `uint8 [0,255]` para `float32 [0.0, 1.0]`.
  - Redes neurais precisam de floats para calcular gradientes (gradientes de inteiros nao existem).
  - Dividir por 255 normaliza para [0,1], escala que facilita a convergencia.
- `.unsqueeze(1)`: adiciona uma dimensao na posicao 1.
  - Antes: `(N, 28, 28)` — N imagens de altura 28 e largura 28.
  - Depois: `(N, 1, 28, 28)` — N imagens com 1 canal, altura 28, largura 28.
  - PyTorch representa imagens no formato **NCHW** (Number of samples, Channels, Height, Width). MNIST e grayscale (escala de cinza), logo C=1. Imagens coloridas teriam C=3 (RGB).

```python
self.labels = torch.tensor(labels, dtype=torch.long)
```

- `dtype=torch.long` (int64): tipo exigido pelo `CrossEntropyLoss`. Rotulos sao indices de classe [0,9], nao floats.

### 2.4. Particionamento dos dados

```
60.000 imagens de treino
    → 54.000 (90%) para treino efetivo
    →  6.000 (10%) para validacao
10.000 imagens → teste
```

**Por que tres conjuntos separados?**

- **Treino**: unico conjunto onde os pesos sao atualizados.
- **Validacao**: avaliado a cada epoca para monitorar overfitting e decidir quando parar. Os pesos NAO sao atualizados com base nele.
- **Teste**: avaliado UMA UNICA VEZ ao final. Nao pode influenciar nenhuma decisao do treinamento ou selecao de hiperparametros.

Usar o conjunto de teste para escolher hiperparametros e **data leakage** (vazamento de dados): o modelo "ve" indiretamente os exemplos de teste durante o desenvolvimento, inflando artificialmente as metricas reportadas.

### 2.5. DataLoader — batching, shuffling e paralelismo

```python
DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
```

- **batch_size=64**: quantas imagens sao processadas de uma vez. O tensor de entrada tera shape `(64, 1, 28, 28)`.
- **shuffle=True** (apenas treino): embaralha a ordem dos exemplos a cada epoch. Isso evita que o SGD veja sempre os mesmos batches na mesma ordem, reduzindo a correlacao entre atualizacoes consecutivas.
- **num_workers=2**: usa 2 processos paralelos para carregar dados em CPU enquanto a GPU calcula o batch anterior. Sobreposicao de I/O e computacao.
- **pin_memory=True**: aloca os tensores em **pinned memory** (memoria paginada/bloqueada) na CPU. Isso permite transferencia assincrona CPU→GPU via DMA (Direct Memory Access), sem envolvimento da CPU.

**Por que shuffle=False na validacao e teste?** A avaliacao deve ser deterministica: o mesmo conjunto de dados, na mesma ordem, sempre. Isso garante que as metricas sejam comparaveis entre diferentes execucoes.

---

## 3. Pre-processamento (preprocessing.py)

### 3.1. Normalizacao z-score (padronizacao)

**Definicao matematica:**

```
x_normalizado = (x - μ) / σ
```

Onde:
- `x`: valor original do pixel em [0, 1] (apos divisao por 255).
- `μ = 0.1307`: media de todos os pixels de todas as imagens do conjunto de treino do MNIST.
- `σ = 0.3081`: desvio padrao de todos os pixels.

**O que isso faz geometricamente?** Translada a distribuicao para media zero e a escala para variancia unitaria. Apos a normalizacao, pixels de fundo ficam em torno de -0.42 (pois 0 → (0 - 0.1307)/0.3081 ≈ -0.42) e pixels de traco ficam em torno de +2.82 (pois 1 → (1 - 0.1307)/0.3081 ≈ 2.82).

**Por que normalizar?**

1. **Gradientes em escala comparavel**: sem normalizacao, pesos conectados a pixels de diferentes escalas recebem gradientes de magnitudes muito diferentes, desbalanceando o aprendizado.
2. **Superficie de perda mais esferica**: gradientes apontam mais diretamente ao minimo em vez de oscilar pelas paredes de um vale estreito.
3. **Compatibilidade com inicializacoes de pesos padrao** (Xavier/He), que assumem entradas com media zero e variancia unitaria.

**Por que usar os valores do treino no teste?** As estatisticas do teste nao podem ser usadas durante o processamento — isso seria data leakage. O modelo e calibrado para as estatisticas do treino, e o teste deve seguir o mesmo pre-processamento para ser uma avaliacao justa.

**No PyTorch:**
```python
T.Normalize(mean=(0.1307,), std=(0.3081,))
```
Isso e aplicado a cada canal de cada imagem: `x_norm[c] = (x[c] - mean[c]) / std[c]`.

### 3.2. Data Augmentation — regularizacao implicita

**Data Augmentation** e a tecnica de gerar versoes transformadas dos dados de treino artificialmente, aumentando a diversidade efetiva sem coletar novos dados reais.

**Rotacao aleatoria (+-10 graus):**
```python
T.RandomRotation(degrees=10)
```
A cada amostra retirada do dataset durante o treino, a imagem e rotacionada por um angulo aleatorio em [-10, +10] graus. Isso simula a variacao natural da inclinacao ao escrever um digito.

**Por que +-10 graus e nao mais?** Uma rotacao de 45 graus poderia fazer um '6' parecer um '9' — a augmentation deve ser realista. Simard et al. (2003) validaram esse intervalo empiricamente para MNIST.

**Translacao aleatoria (+-10%):**
```python
T.RandomAffine(degrees=0, translate=(0.1, 0.1))
```
Desloca a imagem ate 10% da largura/altura em cada eixo (ate ~3 pixels). Simula que o digito nao esta sempre perfeitamente centralizado.

**Por que augmentation so no treino?** A avaliacao mede o desempenho real do modelo em dados "naturais". Se aplicassemos augmentation no teste, estavamos avaliando o modelo em dados artificialmente modificados, nao nos dados reais.

**Mecanismo lazy (sob demanda):** a augmentation e aplicada no `__getitem__` do Dataset, nao na inicializacao. Isso significa que cada vez que o DataLoader solicita a mesma imagem (em epocas diferentes), uma transformacao DIFERENTE e aplicada. O modelo nunca ve exatamente a mesma versao de uma imagem duas vezes.

---

## 4. Arquiteturas de Redes Neurais (architectures.py)

### 4.1. Conceitos Fundamentais

**Camada (Layer)**: unidade computacional que aplica uma transformacao nos dados. Recebe um tensor de entrada, computa uma transformacao (geralmente parametrizada) e produz um tensor de saida.

**nn.Module**: classe base do PyTorch para qualquer componente de rede neural. Define dois metodos essenciais:
- `__init__`: define e registra os sub-modulos e parametros.
- `forward(x)`: define o fluxo computacional (o que acontece quando os dados passam pela camada).

**Parametros treinaveis** (`requires_grad=True`): tensores cujos valores sao ajustados pelo otimizador. O PyTorch rastreia todas as operacoes envolvendo esses tensores para calcular gradientes via autograd.

**`nn.Linear(in_features, out_features)`**: camada totalmente conectada (Fully Connected, FC, ou Dense). Implementa a transformacao afim:
```
y = x @ W^T + b
```
Onde:
- `x`: tensor de entrada de shape `(batch_size, in_features)`.
- `W`: matriz de pesos de shape `(out_features, in_features)` — cada linha e o vetor de pesos de um neuronio.
- `b`: vetor de bias de shape `(out_features,)`.
- `y`: tensor de saida de shape `(batch_size, out_features)`.

**Numero de parametros de uma camada Linear:**
```
parametros = in_features × out_features + out_features
           = out_features × (in_features + 1)   # +1 pelo bias
```
Exemplo: `nn.Linear(784, 512)` tem `784 × 512 + 512 = 401.920` parametros.

**`nn.Flatten()`**: achata todas as dimensoes exceto a do batch em um unico vetor.
- Entrada: `(B, 1, 28, 28)` — B imagens, 1 canal, 28x28 pixels.
- Saida: `(B, 784)` — B vetores de 784 features (1 × 28 × 28 = 784).

**ReLU (Rectified Linear Unit):** funcao de ativacao nao-linear definida como `f(x) = max(0, x)`.
- Para x > 0: gradiente = 1 (nao satura, nao ha vanishing gradient nessa regiao).
- Para x ≤ 0: saida = 0 (neuronio "morto" nesse passo, mas pode reativar em outros batches).
- Computacionalmente eficiente: apenas uma comparacao com zero.
- Cria **esparsidade**: nem todos os neuronios ficam ativos para cada entrada.

**Dropout:** durante o treino, cada neuronio e desativado (zerado) com probabilidade `p` a cada forward pass, de forma independente e aleatoria.
- `nn.Dropout(p=0.25)`: 25% dos neuronios sao zerados em cada passagem.
- Na inferencia (`model.eval()`), dropout e desativado e as ativacoes sao multiplicadas por `(1-p)` para compensar a escala.
- Efeito: o modelo nao pode depender de nenhum neuronio especifico, forcando representacoes redundantes e robustas.

**`nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)`**: camada convolucional 2D. Aplica `out_channels` filtros de tamanho `kernel_size × kernel_size` sobre o tensor de entrada.

- **Filtro/Kernel**: pequena matriz de pesos (ex: 3×3 ou 5×5) que e "deslizada" sobre o feature map de entrada, computando o produto interno em cada posicao.
- **Feature map**: tensor de saida de uma camada convolucional. Cada "canal" do feature map corresponde ao output de um filtro aplicado em todas as posicoes espaciais.
- **Campo receptivo (Receptive Field)**: regiao da entrada original que influencia um neuronio especifico. Um filtro 5×5 tem campo receptivo de 5×5 pixels.
- **Compartilhamento de pesos**: o mesmo filtro e aplicado em TODAS as posicoes espaciais. Isso implica que:
  - A rede detecta o mesmo padrao (ex: borda vertical) independente de onde ele aparece na imagem (equivariancia a translacao).
  - O numero de parametros e `out_channels × in_channels × kernel_H × kernel_W + out_channels (bias)`.

**Formula do tamanho de saida de uma convolucao:**
```
output_size = floor((input_size + 2 × padding - kernel_size) / stride) + 1
```
Exemplo sem padding, stride=1, kernel=5, input=28:
```
output_size = (28 + 0 - 5) / 1 + 1 = 24
```

**Padding**: preenchimento de zeros nas bordas da entrada antes de aplicar o filtro.
- `padding=0`: a saida e menor que a entrada.
- `padding=1` com `kernel_size=3`: mantém o mesmo tamanho espacial.

**MaxPool2d(kernel_size=2, stride=2)**: divide o feature map em regioes 2×2 e seleciona o valor maximo de cada regiao.
- Reduz a dimensao espacial pela metade: `(H, W)` → `(H/2, W/2)`.
- Finalidade: reduz a carga computacional das camadas seguintes e introduz invariancia a pequenas translacoes (se o maximo de uma regiao 2×2 se desloca 1 pixel, o resultado e o mesmo).
- Nao tem parametros treinaveis.

**Batch Normalization (`nn.BatchNorm2d`, `nn.BatchNorm1d`)**: normaliza as ativacoes de um mini-batch, camada por camada.

Para cada canal `c` e cada posicao `(h, w)`:
```
x_norm = (x - μ_batch) / sqrt(σ²_batch + ε)
y = γ × x_norm + β
```
Onde:
- `μ_batch`: media das ativacoes do mini-batch para aquele canal.
- `σ²_batch`: variancia das ativacoes do mini-batch.
- `ε = 1e-5`: constante de estabilidade numerica (evita divisao por zero).
- `γ` e `β`: parametros treinaveis de escala e deslocamento (um par por canal).

Na inferencia, usa-se `running_mean` e `running_var` — medias moveis acumuladas durante o treino.

**Por que BN funciona?** Reduz o "Internal Covariate Shift": a distribuicao dos inputs de cada camada muda a cada atualizacao de pesos das camadas anteriores. Ao normalizar, as camadas posteriores recebem inputs em escala consistente, o que acelera a convergencia e permite learning rates maiores.

**`bias=False` em conv antes de BN**: o BN ja aprende um parametro de deslocamento `β`. Adicionar o bias da conv seria redundante (somaria dois bias que o treinamento nao conseguiria diferenciar). Removendo o bias da conv, economiza-se `out_channels` parametros sem perda de expressividade.

### 4.2. MLP — Multilayer Perceptron

**Notacao e significado de cada componente:**

```
Input(784) → FC(512) → ReLU → Dropout(0.25) → FC(256) → ReLU → Dropout(0.25) → FC(10)
```

**`Input(784)`**:
- O tensor de entrada tem shape `(B, 1, 28, 28)`.
- Apos `nn.Flatten()`, vira `(B, 784)`.
- 784 = 1 canal × 28 pixels de altura × 28 pixels de largura.
- **Problema**: ao achatar, a estrutura espacial e DESTRUIDA. O modelo nao sabe que o pixel na posicao (5,5) e vizinho do pixel (5,6). Trata cada pixel como uma feature independente.

**`FC(512)`** — `nn.Linear(784, 512)`:
- Cada um dos 512 neuronios da camada recebe uma copia do vetor completo de 784 features.
- Cada neuronio aprende uma combinacao linear diferente dos 784 pixels.
- Parametros: 784 × 512 + 512 = **401.920 parametros**.
- Saida: `(B, 512)`.

**`ReLU`** — `F.relu(...)`:
- Aplicado element-wise: `max(0, x)` para cada um dos 512 valores.
- **Necessidade**: sem ativacao nao-linear, empilhar camadas lineares seria equivalente a uma unica transformacao linear. A nao-linearidade permite que a rede aprenda funcoes arbitrariamente complexas (Teorema da Aproximacao Universal — Hornik, 1991).
- Saida: `(B, 512)`, com valores em [0, +∞).

**`Dropout(0.25)`** — `nn.Dropout(p=0.25)`:
- Zera 25% dos 512 valores aleatoriamente a cada forward pass durante o treino.
- Os indices zerados sao sorteados de forma independente para cada amostra no batch.
- Efeito pratico: cada forward pass treina uma "sub-rede" diferente. No conjunto, funciona como um ensemble implicito.
- Saida: `(B, 512)`, com 25% dos valores zerados (durante treino).

**`FC(256)`** — `nn.Linear(512, 256)`:
- Reduz de 512 para 256 neuronios.
- Parametros: 512 × 256 + 256 = **131.328 parametros**.
- Padrao "funil" (512 → 256 → 10): a rede e forcada a comprimir a informacao em representacoes cada vez mais abstratas e compactas.
- Saida: `(B, 256)`.

**`ReLU` e `Dropout(0.25)`**: mesma logica da primeira camada.

**`FC(10)`** — `nn.Linear(256, 10)`:
- 10 neuronios de saida = 10 classes (digitos 0-9).
- Parametros: 256 × 10 + 10 = **2.570 parametros**.
- Saida: `(B, 10)` — 10 **logits**, um por classe.
- **Sem ativacao na ultima camada**: o CrossEntropyLoss aplica o LogSoftmax internamente. Aplicar Softmax antes seria redundante e numericamente menos estavel.

**Total de parametros do MLP**: ~270.000.

**Limitacao fundamental**: MLP nao tem nocao de espaco. O pixel `(0,0)` e o pixel `(27,27)` sao tratados como features completamente independentes. CNNs exploram a estrutura espacial local — e por isso sao superiores para imagens.

### 4.3. LeNet-5 — CNN Classica (LeCun et al., 1998)

**Notacao e significado:**

```
Input(B,1,28,28)
    → Conv(1→6, 5×5) → ReLU → MaxPool(2×2)        → (B,6,12,12)
    → Conv(6→16, 5×5) → ReLU → MaxPool(2×2)        → (B,16,4,4)
    → Flatten                                        → (B,256)
    → FC(256→120) → ReLU → Dropout
    → FC(120→84) → ReLU → Dropout
    → FC(84→10)                                      → (B,10)
```

**`Conv(1→6, 5×5)`** — `nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)`:
- `in_channels=1`: a imagem de entrada tem 1 canal (grayscale).
- `out_channels=6`: 6 filtros independentes, cada um aprende um padrao diferente (ex: bordas horizontais, verticais, diagonais, cantos...).
- `kernel_size=5`: cada filtro tem tamanho 5×5 = 25 pesos + 1 bias = 26 parametros por filtro.
- Total de parametros: `6 × (1 × 5 × 5 + 1) = 156 parametros`.
- **Campo receptivo**: cada valor do feature map de saida depende de uma regiao 5×5 da entrada.
- Sem padding, stride=1: `output_size = (28 - 5) / 1 + 1 = 24`.
- Saida: `(B, 6, 24, 24)` — 6 feature maps de 24×24.

**`MaxPool(2×2)`** — `F.max_pool2d(..., kernel_size=2)`:
- Divide cada feature map em regioes 2×2 e seleciona o maximo.
- `(B, 6, 24, 24)` → `(B, 6, 12, 12)`.
- Sem parametros treinaveis.
- Funcao: reducao de dimensionalidade espacial + invariancia a translacoes.

**`Conv(6→16, 5×5)`** — `nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)`:
- Recebe os 6 feature maps da camada anterior.
- Cada um dos 16 filtros opera SOBRE TODOS OS 6 CANAIS SIMULTANEAMENTE, combinando features de baixo nivel (bordas) em features de alto nivel (partes de digitos).
- Parametros: `16 × (6 × 5 × 5 + 1) = 2.416 parametros`.
- `(B, 6, 12, 12)` → Conv(5×5) → `(B, 16, 8, 8)` → MaxPool → `(B, 16, 4, 4)`.

**`Flatten`**: `(B, 16, 4, 4)` → `(B, 256)`. O 256 vem de `16 × 4 × 4`.

**`FC(256→120)`, `FC(120→84)`, `FC(84→10)`**: classicador final que mapeia as features espaciais extraidas pelas convolucoess para as 10 classes. Os numeros 120 e 84 sao da arquitetura original de 1998.

**Total: ~44.000 parametros — 6x menos que o MLP, com acuracia superior.**

**Por que CNNs sao melhores que MLPs para imagens?**
1. **Compartilhamento de pesos**: o mesmo filtro detecta bordas em qualquer posicao da imagem.
2. **Esparsidade**: cada neuronio convolucional depende de uma regiao local, nao de toda a imagem.
3. **Hierarquia de features**: camadas rasas detectam bordas; camadas profundas detectam formas; camadas ainda mais profundas detectam objetos.

### 4.4. ModernCNN — BatchNorm + Dropout Espacial

**Notacao e significado:**

```
Input(B,1,28,28)
    → Conv(1→32,3×3,pad=1) → BN(32) → ReLU            → (B,32,28,28)
    → Conv(32→32,3×3,pad=1) → BN(32) → ReLU            → (B,32,28,28)
    → MaxPool(2×2) → Dropout2d(0.125)                   → (B,32,14,14)
    → Conv(32→64,3×3,pad=1) → BN(64) → ReLU            → (B,64,14,14)
    → Conv(64→64,3×3,pad=1) → BN(64) → ReLU            → (B,64,14,14)
    → MaxPool(2×2) → Dropout2d(0.125)                   → (B,64,7,7)
    → Flatten                                            → (B,3136)
    → FC(3136→256) → BN1d(256) → ReLU → Dropout(0.25)
    → FC(256→10)                                         → (B,10)
```

**`Conv(1→32, 3×3, pad=1)`**:
- `padding=1` com `kernel_size=3`: o tamanho espacial e PRESERVADO. `output = (28 + 2×1 - 3)/1 + 1 = 28`.
- 32 filtros: mais filtros = mais padroes detectados simultaneamente.
- Parametros por filtro: `3×3×1 = 9` pesos + 1 bias = 10. Total: `32 × 10 = 320`.

**Por que empilhar dois Conv(3×3) em vez de um Conv(5×5)?**
- Campo receptivo equivalente: duas camadas 3×3 consecutivas "enxergam" uma regiao 5×5.
  - Camada 1: cada saida depende de uma regiao 3×3.
  - Camada 2: cada saida depende de 3×3 saidas da camada 1, cada uma dependente de 3×3 pixels originais → campo efetivo = 5×5.
- Menos parametros: `2 × 3^2 = 18` vs. `5^2 = 25`.
- Mais nao-linearidades (duas ReLUs vs. uma): maior poder representacional.

**`BN(32)`** — `nn.BatchNorm2d(32)`:
- Normaliza cada um dos 32 canais usando a media e variancia do mini-batch.
- Aprende 2 parametros por canal (γ e β): total de `32 × 2 = 64` parametros por BN.
- Colocado ANTES do ReLU (convencao padrao).

**`Dropout2d(p=0.125)`** — `nn.Dropout2d(p=dropout_rate/2)`:
- **Dropout espacial (Spatial Dropout)**: desativa feature maps INTEIROS com probabilidade p, nao neuronios individuais.
- Rationale: em dados espaciais, pixels vizinhos sao altamente correlacionados. Desativar um neuronio individual tem pouco impacto porque os vizinhos fornecem informacao similar. Desativar o feature map inteiro forca o modelo a nao depender de padroes detectados por aquele filtro.
- Usa `dropout_rate / 2` (0.125) porque desativar um canal inteiro tem impacto maior que desativar um neuronio.

**`BN1d(256)`** — `nn.BatchNorm1d(256)`:
- Versao do BN para tensores 2D (batch × features), usada em camadas FC.
- Mesmo principio, mas normaliza sobre o eixo do batch para cada feature.

### 4.5. DeepCNN — Blocos Residuais (Inspirado na ResNet)

**O problema que os blocos residuais resolvem — Vanishing Gradient:**

Em redes profundas, o gradiente e calculado pela regra da cadeia:

```
∂L/∂θ_1 = ∂L/∂a_n × ∂a_n/∂a_{n-1} × ... × ∂a_2/∂a_1 × ∂a_1/∂θ_1
```

Se cada fator no produto e < 1 (o que ocorre com sigmoid/tanh), o produto de muitos termos tende a zero exponencialmente. As camadas iniciais recebem gradientes proximo de zero e praticamente nao aprendem.

**A solucao — Skip Connection:**

```python
def forward(self, x):
    residual = self.block(x)  # F(x): duas convolucoess + BN
    out = residual + x        # H(x) = F(x) + x
    return F.relu(out)
```

O gradiente via backpropagation atraves desta operacao:
```
∂L/∂x = ∂L/∂H × (∂F/∂x + 1)
```

O termo "+1" garante que o gradiente nunca se anula, independente de quantas camadas existem. Mesmo que `∂F/∂x ≈ 0` (camada nao contribui), o gradiente ainda flui pela skip connection.

**Notacao detalhada da DeepCNN:**

```
Input(B,1,28,28)

[Stem]
    → Conv(1→32,3×3,pad=1) → BN → ReLU                → (B,32,28,28)

[Stage 1 — ResBlock(32) × 2]
    ResBlock 1:
        x ──────────────────────────────────────────→ (+)
        └→ Conv(32→32,3×3,pad=1,bias=False) → BN → ReLU
           → Conv(32→32,3×3,pad=1,bias=False) → BN ─→ (+) → ReLU
    ResBlock 2: mesma estrutura
    → MaxPool(2×2) → Dropout2d                          → (B,32,14,14)

[Transition]
    → Conv(32→64, 1×1, bias=False) → BN → ReLU         → (B,64,14,14)

[Stage 2 — ResBlock(64) × 2]
    Mesma estrutura, com 64 canais
    → MaxPool(2×2) → Dropout2d                          → (B,64,7,7)

[Classifier]
    → AdaptiveAvgPool2d(1)                               → (B,64,1,1)
    → Flatten                                            → (B,64)
    → FC(64→128) → BN1d → ReLU → Dropout
    → FC(128→10)                                         → (B,10)
```

**`Stem`**: primeira convolucao que extrai features iniciais da imagem bruta. "Caule" da rede — ponto de entrada antes dos blocos residuais.

**`ResidualBlock(channels)`**:
- Recebe e produz tensores com o MESMO numero de canais e MESMAS dimensoes espaciais.
- A skip connection `H(x) = F(x) + x` requer que `F(x)` e `x` tenham o mesmo shape (adicao element-wise).
- `bias=False` nas convolucoess porque BN ja inclui parametro de deslocamento β.

**`Transition` — Convolucao 1×1 (`Conv(32→64, kernel_size=1)`)**:
- Convolucao com kernel 1×1 opera em cada posicao espacial independentemente, combinando os 32 canais em 64. Nao altera as dimensoes espaciais (H, W).
- Necessaria porque os blocos residuais exigem `in_channels == out_channels`. Para mudar o numero de canais entre estagios, usa-se essa "convolucao pontual" (pointwise convolution).

**`AdaptiveAvgPool2d(1)`**: calcula a media global de cada feature map.
- Entrada: `(B, 64, 7, 7)`.
- Saida: `(B, 64, 1, 1)`.
- Cada um dos 64 valores e a media de todos os 49 pixels daquele feature map.
- Equivalente ao Global Average Pooling (GAP) de Lin et al. (2014).
- Beneficios: elimina a dependencia do tamanho espacial da entrada (pode usar imagens de qualquer tamanho) e atua como regularizador forte, forcando os feature maps a conter informacao global.

**Total: 10 camadas convolucionais, ~125.000 parametros.**

### 4.6. Factory Pattern — selecao dinamica de arquitetura

```python
ARCHITECTURE_REGISTRY = {
    'MLP': MLP, 'LeNet5': LeNet5, 'ModernCNN': ModernCNN, 'DeepCNN': DeepCNN
}

def build_model(architecture: str, dropout_rate: float) -> nn.Module:
    model_class = ARCHITECTURE_REGISTRY[architecture]
    return model_class(dropout_rate=dropout_rate)
```

O **Registry Pattern** mapeia strings a classes. A funcao `build_model` instancia o modelo dinamicamente pelo nome, sem `if/elif` duplicados. Isso permite que o tuning itere sobre arquiteturas programaticamente (`trial.suggest_categorical('architecture', ['MLP', 'LeNet5', ...])`).

---

## 5. Treinamento (training.py)

### 5.1. O Ciclo Fundamental de Aprendizado Supervisionado

Para cada mini-batch `(x, y)` onde `x` sao as imagens e `y` os rotulos:

**Passo 1 — Zero Gradients:**
```python
optimizer.zero_grad(set_to_none=True)
```
O PyTorch ACUMULA gradientes por padrao (soma os gradientes de todas as chamadas `.backward()`). Se nao zeramos antes de cada batch, o gradiente do batch atual seria somado ao do batch anterior — o que seria incorreto para SGD padrao. `set_to_none=True` e mais eficiente que preencher com zeros pois desaloca os tensores de gradiente.

**Passo 2 — Forward Pass:**
```python
outputs = model(images)  # shape: (B, 10)
```
Os dados fluem da entrada ate a saida. O PyTorch automaticamente constroi o **grafo computacional** (computational graph) — um DAG (Directed Acyclic Graph) onde os nos sao tensores e as arestas sao operacoes. Esse grafo e necessario para o backward pass.

**Passo 3 — Calculo da Perda (Loss):**
```python
loss = criterion(outputs, labels)
```
`criterion` e `nn.CrossEntropyLoss()`. Internamente:
1. **LogSoftmax**: `log_softmax(x_i) = x_i - log(Σ_j exp(x_j))`.
   Versao numericamente estavel do `log(softmax(x))`. Converte logits em log-probabilidades.
2. **NLLLoss (Negative Log-Likelihood)**: `-log_softmax(x_{y_true})`.
   Penaliza a log-probabilidade atribuida a classe correta.

Resultado: escalar representando a perda media do mini-batch. Quanto menor, mais o modelo concorda com os rotulos verdadeiros.

**Por que nao MSE?** MSE = `(ŷ - y)²`. Para classificacao:
- Com `y ∈ {0,1}` (one-hot), o gradiente do MSE satura quando a predicao e muito errada (ex: predicao = 0, verdade = 1 → gradiente = 2, pequeno).
- CrossEntropy tem gradiente `softmax(x) - y`, que e grande quando a predicao e muito errada e pequeno quando e quase correta. Comportamento ideal para aprendizado.

**Passo 4 — Backward Pass (Backpropagation):**
```python
loss.backward()
```
Percorre o grafo computacional de tras para frente, calculando `∂L/∂θ` para cada parametro `θ` com `requires_grad=True` usando a regra da cadeia. Os gradientes sao armazenados em `param.grad` para cada parametro.

**Passo 5 — Atualizacao dos Pesos:**
```python
optimizer.step()
```
Usa os gradientes em `param.grad` para atualizar cada parametro segundo a regra do otimizador.

### 5.2. Otimizadores — como os pesos sao atualizados

**SGD com Momentum (Polyak, 1964) e Nesterov:**

Equacao padrao do SGD:
```
θ_t = θ_{t-1} - lr × g_t
```
Onde `g_t = ∂L/∂θ` e o gradiente no batch atual.

Com **Momentum** (μ = 0.9):
```
v_t = μ × v_{t-1} + g_t           (acumulacao da "velocidade")
θ_t = θ_{t-1} - lr × v_t
```
O momentum acumula gradientes passados. Em direcoes consistentes, v cresce (aceleracao). Em direcoes oscilatories, v se cancela (amortecimento). Ajuda a atravessar "platôs" e "vales estreitos" na superficie de perda.

Com **Nesterov** (Nesterov, 1983):
```
θ_lookahead = θ_{t-1} - lr × μ × v_{t-1}   (posicao "futura" estimada)
g_t = ∂L/∂θ_lookahead                        (gradiente na posicao futura)
v_t = μ × v_{t-1} + g_t
θ_t = θ_{t-1} - lr × v_t
```
Calcula o gradiente na posicao que o momentum levaria, em vez da posicao atual. Convergencia mais rapida por ser um "look-ahead".

**Adam (Kingma & Ba, 2015):**
```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t          (1o momento: media exponencial)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²          (2o momento: variancia)
m̂_t = m_t / (1 - β₁^t)                         (correcao de bias inicial)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)
```
Parametros: β₁=0.9, β₂=0.999, ε=1e-8.

O **1o momento** (m) e como o momentum do SGD — suaviza o gradiente. O **2o momento** (v) estima a variancia do gradiente por parametro. A divisao `m̂/√v̂` **adapta a taxa de aprendizado para cada parametro individualmente**: parametros com gradientes consistentemente altos recebem lr efetiva menor; parametros com gradientes esparsos recebem lr efetiva maior.

**AdamW (Loshchilov & Hutter, 2019):**
No Adam original, weight decay (L2) e implementado adicionando `λ × θ` ao gradiente: `g_t' = g_t + λ × θ`. Mas isso interage com a normalizacao adaptativa do Adam de forma incorreta — parametros com gradientes pequenos (logo lr efetiva alta) recebem mais penalizacao proporcional.

O AdamW "desacopla" o weight decay, aplicando-o diretamente nos pesos:
```
θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε) - lr × λ × θ_{t-1}
```
O termo `- lr × λ × θ_{t-1}` e independente do Adam, resultando em regularizacao mais efetiva e melhor generalizacao.

### 5.3. Schedulers de Learning Rate

**Por que ajustar a lr durante o treino?**
- **Fase inicial**: lr alta para explorar o espaco de parametros rapidamente.
- **Fase final**: lr baixa para convergir com precisao a um minimo.

**StepLR**:
```
lr_t = lr_0 × γ^(floor(epoch / step_size))
```
Com `step_size = epochs//3` e `γ = 0.1`: divide a lr por 10 a cada terco do treino. Simples, deterministic e interpretavel.

**CosineAnnealingLR** (Loshchilov & Hutter, 2017):
```
lr_t = η_min + 0.5 × (η_max - η_min) × (1 + cos(π × t / T))
```
Onde `t` e a epoca atual e `T` o numero total de epocas. A lr cai suavemente seguindo uma curva cosseno de η_max ate η_min ≈ 0. Sem degraus abruptos; a curva suave ajuda a convergir para minimos mais agudos (que podem generalizar melhor).

**ReduceLROnPlateau**: monitora a `val_loss`. Se nao melhorar por `patience=5` epocas, multiplica a lr por `factor=0.5`. Abordagem reativa — so atua quando necessario.

### 5.4. Classe Trainer — Early Stopping e Checkpointing

**Early Stopping** (Prechelt, 1998):

O `Trainer` monitora `val_loss` a cada epoca. Se nao melhorar por `patience=7` epocas consecutivas:
```python
if val_metrics['loss'] < self.best_val_loss:
    self.best_val_loss = val_metrics['loss']
    self.epochs_no_improve = 0
    self.best_model_state = copy.deepcopy(self.model.state_dict())
    torch.save(checkpoint, self.checkpoint_path)
else:
    self.epochs_no_improve += 1
    if self.epochs_no_improve >= self.patience:
        break  # Interrompe o treinamento
```

**`state_dict()`**: dicionario que mapeia nomes de camadas a seus tensores de pesos. E a representacao serializada do modelo.

**`copy.deepcopy`**: necessario porque os pesos sao tensores mutaveis. Uma referencia simples apontaria para os pesos atuais (que continuariam mudando). O deepcopy salva uma copia independente dos pesos no momento do melhor desempenho.

**Checkpoint salvo em disco:**
```python
torch.save({
    'epoch': epoch,
    'model_state': self.best_model_state,
    'optimizer_state': self.optimizer.state_dict(),
    'val_loss': self.best_val_loss,
    'val_acc': self.best_val_acc,
}, self.checkpoint_path)
```

Ao final do treinamento, os pesos do MELHOR epoch sao restaurados:
```python
self.model.load_state_dict(self.best_model_state)
```
Garantia: o modelo retornado e o da melhor generalizacao, nao o da ultima epoca.

**Integracao com Optuna — Pruning:**
```python
trial.report(val_metrics['accuracy'], epoch)
if trial.should_prune():
    raise optuna.exceptions.TrialPruned()
```
A cada epoca, o trainer reporta a metrica ao Optuna. Se o Optuna decide que o trial deve ser podado (performance muito abaixo da mediana na mesma epoca), uma excecao especial e levantada, interrompendo o treinamento precocemente.

---

## 6. Avaliacao (evaluation.py)

### 6.1. Coleta de predicoes

```python
probabilities = torch.softmax(outputs, dim=1)
_, predicted = torch.max(outputs, dim=1)
```

**Softmax**: `softmax(x_i) = exp(x_i) / Σ_j exp(x_j)`.
Normaliza os 10 logits em probabilidades que somam 1. A probabilidade do classe `i` representa a "confianca" do modelo de que a imagem pertence a classe `i`.

**`torch.max(outputs, dim=1)`**: retorna o valor maximo e o INDICE do maximo ao longo da dimensao 1 (classes). O indice e a classe predita (argmax). Notar que o argmax dos logits e identico ao argmax das probabilidades softmax (pois exp e monotonica).

### 6.2. Metricas de classificacao

Para cada classe `c` definimos:
- **TP (True Positive)**: exemplos da classe c corretamente preditos como c.
- **FP (False Positive)**: exemplos de outras classes incorretamente preditos como c.
- **FN (False Negative)**: exemplos da classe c incorretamente preditos como outra classe.
- **TN (True Negative)**: exemplos de outras classes corretamente preditos como nao-c.

**Precision** (Precisao):
```
P_c = TP_c / (TP_c + FP_c)
```
"Das vezes que o modelo disse que era o digito c, em quantas ele estava certo?"
Penaliza falsos positivos. Importante quando o custo de um alarme falso e alto.

**Recall** (Revocacao / Sensibilidade):
```
R_c = TP_c / (TP_c + FN_c)
```
"De todos os exemplos reais do digito c, quantos o modelo detectou?"
Penaliza falsos negativos. Importante quando o custo de uma deteccao perdida e alto.

**F1-Score**:
```
F1_c = 2 × P_c × R_c / (P_c + R_c)
```
Media harmonica de Precision e Recall. A media harmonica penaliza desbalanceamento: um modelo com P=1.0 e R=0.1 teria F1 = 0.18, revelando o problema.

**Por que media harmonica e nao aritmetica?** A media aritmetica de P=1.0 e R=0.0 seria 0.5, sugerindo desempenho razoavel. A media harmonica seria 0.0, revelando que o modelo e inutilizavel.

**Macro average**: media simples entre as F1 de todas as 10 classes. Trata todas as classes igualmente.
**Weighted average**: media ponderada pelo numero de exemplos de cada classe (support). Para datasets balanceados como MNIST, coincide quase completamente com a macro.

### 6.3. Matriz de confusao

```
Matriz C de tamanho 10×10 onde:
C[i][j] = numero de exemplos da classe real i preditos como classe j
```

A diagonal `C[i][i]` contem os True Positives de cada classe. Elementos fora da diagonal sao erros. Padroes sistematicos fora da diagonal revelam confusoes do modelo (ex: C[4][9] alto significa que o modelo frequentemente confunde '4' com '9').

Normalizada por linha: `C_norm[i][j] = C[i][j] / sum(C[i,:])`. Cada linha soma 1. O elemento `C_norm[i][i]` e o Recall da classe i.

### 6.4. Curvas de aprendizado

Plotagem de loss e accuracy por epoca, para treino e validacao. Diagnosticos:

| Padrao | Interpretacao |
|--------|--------------|
| Treino e val diminuindo juntas | Aprendizado saudavel |
| Treino diminui, val estagna/sobe | Overfitting |
| Ambas altas | Underfitting (modelo simples ou lr muito baixa) |
| Treino e val muito prooximas | Boa generalizacao |
| Val explode | Learning rate muito alta, instabilidade |

---

## 7. Otimizacao de Hiperparametros (tuning.py)

### 7.1. O problema de otimizacao de hiperparametros

Dado um espaco de hiperparametros H (8 dimensoes no nosso caso) e uma funcao de avaliacao `f(h) = val_accuracy` (cara de computar — requer treinar um modelo completo), queremos encontrar:

```
h* = argmax f(h)
         h ∈ H
```

### 7.2. Funcao objetivo do Optuna

O Optuna chama `objective(trial)` N vezes. Cada chamada:
1. **Sugere** hiperparametros: `trial.suggest_*()`.
2. **Treina** um modelo completo com esses hiperparametros.
3. **Retorna** a melhor acuracia de validacao atingida.

O Optuna usa os resultados para guiar as proximas sugestoes (no caso do TPE).

**`trial.suggest_categorical('architecture', ['MLP', 'LeNet5', ...])`**: sugere um valor de uma lista discreta.

**`trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)`**: sugere um float em escala log-uniforme. Com `log=True`, a probabilidade de sugerir um valor entre 1e-4 e 1e-3 e igual a de sugerir entre 1e-2 e 1e-1 — adequado para lr, onde o impacto e multiplicativo.

**`trial.set_user_attr('num_parameters', num_params)`**: armazena informacoes adicionais no trial, acessiveis depois como `trial.user_attrs['num_parameters']`. Usado pelo modulo de persistencia para comparar complexidade dos modelos sem precisar reinstaciá-los.

### 7.3. TPE — Tree-structured Parzen Estimator (Bergstra et al., 2011)

**Principio da otimizacao bayesiana**: em vez de testar hiperparametros aleatoriamente, constroi-se um modelo probabilistico (surrogate model) da funcao objetivo e usa-se esse modelo para decidir qual ponto testar a seguir.

O TPE constroi dois modelos de densidade kernel (KDE — Kernel Density Estimation):
- `l(x)`: densidade dos hiperparametros nos "bons" trials (top γ%, ex: top 25%).
- `g(x)`: densidade dos hiperparametros nos "maus" trials (restante).

O proximo ponto e escolhido maximizando a funcao de aquisicao:
```
EI(x) ∝ l(x) / g(x)
```
Isso seleciona hiperparametros que sao frequentes nos bons trials e raros nos maus — concentrando a busca nas regioes promissoras.

**`n_startup_trials=10`**: os primeiros 10 trials sao puramente aleatorios. Sao necessarios para construir os modelos `l(x)` e `g(x)` com dados suficientes antes de comecar a guiar a busca.

### 7.4. Random Search (Bergstra & Bengio, 2012)

Amostra hiperparametros aleatoriamente. Surpreendentemente eficaz porque:
- Cada trial explora uma combinacao UNICA de todos os hiperparametros.
- Grid Search, ao contrario, repete valores de hiperparametros "irrelevantes": se a lr nao importa muito mas o otimizador importa, Grid Search testa cada combinacao (lr × otimizador), desperdicando trials em variacoes de lr que tem o mesmo efeito.

### 7.5. Grid Search

Avalia TODAS as combinacoes de um grid discreto predefinido. Para o nosso espaco:
```
4 arquiteturas × 3 lr × 4 batch_size × 3 otimizadores × 3 epocas × 3 dropout × 3 wd × 4 schedulers
= 4 × 3 × 4 × 3 × 3 × 3 × 3 × 4 = 15.552 trials
```
Impraticavel para espacos grandes. Util apenas quando o espaco e pequeno e se quer garantia de cobertura total.

### 7.6. MedianPruner — poda de trials

```python
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
```

A cada epoca, o Optuna verifica se o trial atual esta abaixo da **mediana** dos trials anteriores na mesma epoca. Se estiver, o trial e **podado** (interrompido precocemente), economizando o tempo que seria gasto nas epocas restantes.

- `n_startup_trials=5`: espera 5 trials completos antes de comecar a podar (sem mediana com poucos dados).
- `n_warmup_steps=5`: nao poda nas primeiras 5 epocas de cada trial (redes precisam de tempo para comecar a aprender).
- `interval_steps=1`: verifica a cada epoca.

---

## 8. Sistema de Persistencia (persistence.py)

### 8.1. Arquivos gerados

**`results/best_params.yaml`**: YAML com a melhor configuracao de hiperparametros ja encontrada. Estrutura:
```yaml
hyperparameters:
  architecture: LeNet5
  learning_rate: 0.001
  batch_size: 64
  optimizer: Adam
  epochs: 20
  dropout_rate: 0.25
  weight_decay: 0.0001
  scheduler: CosineAnnealingLR
metrics:
  val_accuracy: 0.9923
  num_parameters: 44426
  training_time_seconds: 142.3
metadata:
  sampler: tpe
  n_trials: 50
  timestamp: "2026-03-24 14:30:00"
```

**`results/tuning_history.yaml`**: lista YAML com TODAS as execucoes de tuning, incluindo as que nao se tornaram o melhor resultado. Cada entrada tem um campo `was_saved_as_best: true/false`.

### 8.2. Resolucao de hiperparametros — 3 niveis de prioridade

```
YAML defaults  →  best_params.yaml  →  argumentos CLI
  (menor)                              (maior)
```

Em codigo:
```python
defaults = config['defaults'].copy()          # Nivel 1: YAML

if not args.no_saved_params:
    saved_best = load_best_params(args.output_dir)
    if saved_best is not None:
        defaults.update(saved_best['hyperparameters'])  # Nivel 2: tuning

architecture = args.architecture or defaults['architecture']  # Nivel 3: CLI
```

O metodo `dict.update()` sobrescreve as chaves do nivel anterior. Argumentos CLI tem prioridade maxima porque sao verificados por ultimo (`args.X or defaults['X']`).

### 8.3. Comparacao entre tunings — criterios hierarquicos

```python
ACCURACY_THRESHOLD = 0.0001  # 0.01%
```

**Criterio 1 — Acuracia de validacao:**
```python
if new_acc > old_acc + ACCURACY_THRESHOLD:
    return True  # Novo e melhor
if old_acc > new_acc + ACCURACY_THRESHOLD:
    return False  # Antigo e melhor
```
O threshold de 0.01% evita que flutuacoes de ponto flutuante (que podem ocorrer em diferentes execucoes pelo nao-determinismo de operacoes paralelas em GPU) causem trocas desnecessarias.

**Criterio 2 — Numero de parametros (Navalha de Occam):**
```python
if new_params < old_params:
    return True  # Modelo mais simples com mesma acuracia e preferido
```
Principio: entre dois modelos com desempenho equivalente, o mais simples generaliza melhor (menos propenso a overfitting) e e mais eficiente computacionalmente.

**Criterio 3 — Tempo de treinamento:**
```python
return new_time < old_time  # Menor tempo e preferido
```
Desempate final por eficiencia.

---

## 9. Reproducibilidade

### 9.1. Fixacao de sementes — por que cada uma e necessaria

```python
random.seed(seed)           # Usado em data augmentation (RandomRotation, RandomAffine)
np.random.seed(seed)        # Usado em operacoes de array e shuffling via NumPy
torch.manual_seed(seed)     # Inicializacao de pesos em CPU e shuffling do DataLoader
torch.cuda.manual_seed_all(seed)   # Inicializacao em GPU (se disponivel)
torch.backends.cudnn.deterministic = True   # Forca algoritmos deterministicos
torch.backends.cudnn.benchmark = False      # Desabilita selecao automatica de algoritmos
```

**Por que `cudnn.deterministic = True`?** O cuDNN tem multiplas implementacoes de cada operacao (ex: convolucao). O modo `benchmark=True` testa varias e escolhe a mais rapida para o hardware atual — mas resultados podem variar entre execucoes porque a selecao pode depender do estado interno. `deterministic=True` forca sempre o mesmo algoritmo.

**Trade-off**: `deterministic=True` pode ser 10-20% mais lento que o modo nao-deterministico.

### 9.2. Semente 42 — convencao da comunidade

A semente 42 e uma convencao amplamente usada na comunidade de ML (referencia cultural a "A Resposta para a Vida, o Universo e Tudo Mais" do livro O Guia do Mochileiro das Galaxias). Nao tem propriedade especial — qualquer inteiro fixo funcionaria.

---

## 10. Conceitos Teoricos Fundamentais

### 10.1. Backpropagation — como os gradientes sao calculados

Para uma rede com L camadas, a saida e `ŷ = f_L(f_{L-1}(...f_1(x)))`. A perda e `L = loss(ŷ, y)`.

O gradiente do parametro `θ_k` da camada k e:
```
∂L/∂θ_k = ∂L/∂a_L × ∂a_L/∂a_{L-1} × ... × ∂a_{k+1}/∂a_k × ∂a_k/∂θ_k
```
Onde `a_i` e a ativacao da camada i.

O PyTorch calcula isso automaticamente via **autograd**: durante o forward pass, cada operacao registra como calcular seu gradiente. Durante o backward pass, esses calculos sao executados em ordem reversa.

**Problema do Vanishing Gradient**: se cada `∂a_i/∂a_{i-1} < 1`, o produto de L termos tende exponencialmente a zero. Solucoes: ReLU, BatchNorm, residual connections.

**Problema do Exploding Gradient**: se cada termo > 1, o produto explode. Solucao: gradient clipping (nao implementado neste projeto, mas relevante para RNNs).

### 10.2. CrossEntropyLoss — por que e a perda correta para classificacao

**Motivacao probabilistica**: assumimos que o modelo parametriza uma distribuicao categorica sobre as 10 classes. A perda CrossEntropy maximiza a **verossimilhanca** dos dados observados sob essa distribuicao — criterio fundamental da estatistica (estimacao por maxima verossimilhanca, MLE).

```
L = -log P(y | x; θ) = -log softmax(f_θ(x))_y = -f_θ(x)_y + log(Σ_j exp(f_θ(x)_j))
```

**Gradiente**: `∂L/∂f_i = softmax(f)_i - 1[i=y]`.
- Para a classe correta (i=y): gradiente = `softmax(f)_y - 1`. Se o modelo e muito confiante (softmax ≈ 1), gradiente ≈ 0 (convergiu). Se errado (softmax ≈ 0), gradiente ≈ -1 (forte sinal de correcao).
- Para classes erradas (i≠y): gradiente = `softmax(f)_i`. Penaliza probabilidades altas atribuidas a classes erradas.

### 10.3. Por que pesos em float32 e nao float64?

- float32 (32 bits) tem precisao de ~7 digitos decimais, suficiente para gradientes de redes neurais.
- float64 (64 bits) usaria 2x mais memoria e seria 2x mais lento em operacoes matriciais nas GPUs (que sao otimizadas para float32).
- Mixed precision (float16 para forward, float32 para pesos) e possivel mas nao implementado neste projeto.

### 10.4. NCHW — formato de tensores no PyTorch

**N**: numero de amostras no batch.
**C**: numero de canais (1 para grayscale, 3 para RGB).
**H**: altura em pixels.
**W**: largura em pixels.

PyTorch usa NCHW por padrao. As operacoes de convolucao (`nn.Conv2d`) esperam especificamente esse formato. TensorFlow usa NHWC por padrao — uma diferenca de implementacao importante ao portar codigo entre frameworks.

---

## 11. Artefatos Gerados

| Artefato | Caminho | Descricao |
|----------|---------|-----------|
| Pesos do modelo | `results/checkpoints/{arch}_best.pt` | state_dict com os pesos do melhor epoch |
| Curvas de aprendizado | `results/figures/{arch}_learning_curves.png` | Loss e accuracy por epoca |
| Matriz de confusao | `results/figures/{arch}_confusion_matrix.png` | Heatmap 10×10 normalizado |
| Exemplos de erros | `results/figures/{arch}_misclassified.png` | Grade de predicoes incorretas |
| Relatorio de metricas | `results/{arch}_report.txt` | Precision, Recall, F1 por classe |
| Relatorio de tuning | `results/tuning_report.txt` | Top 10 trials com hiperparametros |
| Melhores parametros | `results/best_params.yaml` | Hiperparametros para reutilizacao |
| Historico de tunings | `results/tuning_history.yaml` | Log cumulativo de todos os tunings |
| Log do experimento | `results/logs/experiment.log` | Log completo com timestamps |

---

## 12. Comandos e Exemplos de Uso

```bash
# Treinamento com defaults do YAML (LeNet5, Adam, lr=1e-3, 20 epocas)
python main.py --mode train

# Treinamento com arquitetura especifica, sobrescrevendo o YAML
python main.py --mode train --architecture DeepCNN --epochs 30 --learning_rate 0.0005

# Tuning com 50 trials usando TPE (busca bayesiana)
python main.py --mode tune --n_trials 50 --sampler tpe

# Tuning com Random Search em GPU
python main.py --mode tune --n_trials 30 --sampler random --device cuda

# Treinar usando automaticamente os melhores hiperparametros do tuning anterior
python main.py --mode train

# Treinar ignorando os resultados do tuning (usa apenas defaults do YAML)
python main.py --mode train --no_saved_params

# Treinar com a arquitetura do CLI mas demais hiperparametros do tuning
python main.py --mode train --architecture DeepCNN
```

---

## 13. Referencias Bibliograficas

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. arXiv:1512.03385.
3. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*.
4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15, 1929-1958.
5. Kingma, D. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
6. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
7. Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*.
8. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD*.
9. Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR*, 13, 281-305.
10. Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011). "Algorithms for Hyper-Parameter Optimization." *NeurIPS*.
11. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.
12. Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.
13. Simard, P., Steinkraus, D., & Platt, J. (2003). "Best Practices for CNNs Applied to Visual Document Analysis." *ICDAR*.
14. Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533-536.
15. Simonyan, K. & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
16. Prechelt, L. (1998). "Early Stopping — But When?" In *Neural Networks: Tricks of the Trade*.
17. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning." Springer.
18. Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). "On the Importance of Initialization and Momentum in Deep Learning." *ICML*.
19. Nesterov, Y. (1983). "A Method for Solving the Convex Programming Problem with Convergence Rate O(1/k²)." *Soviet Mathematics Doklady*.
20. Hornik, K. (1991). "Approximation Capabilities of Multilayer Feedforward Networks." *Neural Networks*, 4(2), 251-257.
21. Lin, M., Chen, Q., & Yan, S. (2014). "Network In Network." *ICLR*.
