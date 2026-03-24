# Explicacao Completa da Solucao — MNIST Academic Baseline

---

## Resumo Rápido para Apresentação (Elevator Pitch)

1. **Leitura Binária de Baixo Nível (Parsing manual):** Em vez de usar bibliotecas prontas como o `torchvision`, implementei a leitura direta dos arquivos originais `.idx` usando a biblioteca `struct`. Li byte a byte, converti para arrays NumPy (`np.frombuffer`) e montei os DataLoaders do PyTorch. Isso prova o entendimento real do formato de dados.
2. **Pré-Processamento Rigoroso:** Separei os dados estritamente em Treino, Validação e Teste. Apliquei normalização (Z-Score) usando a média e desvio padrão *exclusivos* do treino (evitando data leakage) e integrei Data Augmentation no próprio pipeline para prevenir overfitting.
3. **Progressão de 4 Arquiteturas (Feitas do zero):** Ao invés de usar uma rede genérica, criei uma esteira de evolução histórica na visão computacional:
   - **MLP**: Baseline simples e raso para comparação inicial.
   - **LeNet-5**: Introdução das primeiras convoluções locais reais.
   - **ModernCNN**: Evolução usando Batch Normalization e Spatial Dropout (Dropout2d).
   - **DeepCNN**: Refinamento de ponta utilizando blocos residuais (skip connections) para não perder gradiente e Global Average Pooling para classificação dinâmica.
4. **Treinamento e Engenharia:** Construí o loop de treinamento profissional abstraindo early stopping monitorando corretamente a `validation_loss` e não a de teste. Integrei otimizadores como AdamW em conjunto com agendadores modernos de learning rate (como Cosine Annealing LR).
5. **Tuning de Hiperparâmetros (Optuna):** Não houve tentativa e erro manual; embuti uma camada de tuning com busca bayesiana (TPE) do Optuna. Mais do que isso, desenvolvi um sistema de persistência onde o pipeline sabe decidir entre múltiplas execuções se o modelo atual é estritamente melhor, armazenando o vencedor e seus parâmetros em um `best_params.yaml`.
6. **Rigor e Reprodutibilidade Absoluta:** Congelei as sementes (random seeds) estocásticas em todos os ecossistemas: built-in do Python, NumPy, PyTorch de CPU genérico e determinismo da engine do CUDA da GPU. 

---

## PARTE 1 — Nivelamento: Entendendo os Conceitos Base

Antes de qualquer detalhe tecnico, e preciso entender o que cada palavra significa. Esta secao explica os termos do zero.

---

### O que e o MNIST?

MNIST e um banco de dados de 70.000 imagens de digitos manuscritos (0 a 9), criado por LeCun et al. em 1998. Cada imagem e um quadrado de **28x28 pixels** em escala de cinza (preto e branco). A tarefa e: dado uma imagem, dizer qual digito ela representa. Isso e um problema de **classificacao** — a resposta e uma entre 10 categorias fixas.

---

### O que e um Pixel?

Um pixel e o menor elemento de uma imagem. Em escala de cinza, cada pixel e representado por um numero inteiro de **0 a 255**:
- 0 = preto absoluto
- 255 = branco absoluto
- 127 = cinza medio

Uma imagem 28x28 tem `28 × 28 = 784 pixels` no total.

---

### O que e uma Rede Neural?

Uma rede neural e um programa que **aprende a partir de exemplos**. Voce nao programa as regras manualmente — voce mostra milhares de exemplos (imagem + rotulo correto) e a rede ajusta seus parametros internos ate conseguir reconhecer os padroes sozinha.

A analogia mais simples: e como ensinar uma crianca a reconhecer cachorros mostrando fotos e dizendo "isso e cachorro" ou "isso nao e cachorro". A crianca nao aprende uma regra explicita — ela aprende pelos exemplos.

---

### O que sao Parametros (Pesos)?

Os **parametros** (tambem chamados de **pesos**) sao os numeros internos da rede neural que sao ajustados durante o treinamento. Pense neles como os "botoes de controle" da rede.

Uma rede com 44.000 parametros tem 44.000 numeros que sao ajustados para que a rede erre cada vez menos.

---

### O que e um Tensor?

Um **tensor** e simplesmente uma estrutura de dados com multiplas dimensoes:
- **Escalar** (0D): um unico numero. Ex: `5.3`
- **Vetor** (1D): uma lista de numeros. Ex: `[1, 2, 3, 4]`
- **Matriz** (2D): uma tabela de numeros. Ex: uma imagem 28x28
- **Tensor 3D**: uma pilha de matrizes. Ex: 64 imagens 28x28 = tensor de shape `(64, 28, 28)`
- **Tensor 4D**: shape `(64, 1, 28, 28)` = 64 imagens, 1 canal de cor, 28 altura, 28 largura

O PyTorch trabalha exclusivamente com tensores.

---

### O que e Shape?

**Shape** e a forma de um tensor — quantos elementos ele tem em cada dimensao.

Exemplos:
- Uma unica imagem MNIST: shape `(1, 28, 28)` — 1 canal, 28 linhas, 28 colunas
- Um batch de 64 imagens: shape `(64, 1, 28, 28)`
- O vetor de saida da rede (10 classes): shape `(64, 10)` — 64 predicoes, cada uma com 10 valores

---

### O que e Batch (Mini-batch)?

Processar todas as 54.000 imagens de uma vez seria muito pesado para a memoria. Processar uma por vez seria muito lento. O **batch** e o meio-termo: processa-se um grupo de imagens de cada vez.

`batch_size = 64` significa que a rede processa 64 imagens simultaneamente, calcula o erro medio dessas 64, e ajusta os pesos uma vez.

---

### O que e uma Epoca (Epoch)?

Uma **epoca** e uma passagem completa por todo o conjunto de treino. Se o treino tem 54.000 imagens e o batch_size e 64, uma epoca tem `ceil(54000 / 64) = 844` iteracoes.

Treinar por 20 epocas significa que cada imagem e vista pela rede 20 vezes.

---

### O que e Forward Pass e Backward Pass?

- **Forward Pass**: os dados entram na rede, passam por todas as camadas, e saem como uma predicao. Sentido: entrada → saida.
- **Backward Pass**: o erro da predicao e calculado e propagado de volta pela rede, calculando o quanto cada parametro contribuiu para o erro. Sentido: saida → entrada.

Analogia: voce estuda (forward pass), erra uma questao, entende QUAL parte do seu raciocinio estava errada (backward pass), e corrige.

---

### O que e Gradiente?

O **gradiente** e um vetor que indica, para cada parametro da rede, em qual direcao e o quanto esse parametro deve mudar para reduzir o erro.

Analogia: imagine que voce esta numa montanha com os olhos fechados e quer descer. O gradiente e a inclinacao do chao sob seus pes — ele diz em qual direcao o chao sobe mais steeply, entao voce anda na direcao oposta.

---

### O que e Learning Rate (Taxa de Aprendizado)?

A **learning rate** controla o tamanho do passo que o otimizador da ao atualizar os pesos.

- **lr muito alta**: os passos sao grandes demais e a rede "pula" sobre o minimo, oscilando ou divergindo.
- **lr muito baixa**: os passos sao pequenos demais e o treino fica lentissimo.
- **lr ideal**: converge de forma estavel e rapida.

Valores tipicos: entre `0.0001` e `0.01`.

---

### O que e Overfitting e Underfitting?

- **Overfitting**: a rede "decora" os dados de treino mas nao generaliza. Como um aluno que memoriza as respostas do simulado mas nao entende a materia — vai mal na prova real.
- **Underfitting**: a rede e simples demais para aprender os padroes. Como um aluno que nao estudou nada.
- **Ideal**: a rede aprende os padroes gerais e se sai bem em dados novos.

Como detectar: se a acuracia no treino e muito maior que na validacao, e overfitting. Se ambas sao baixas, e underfitting.

---

### O que e Regularizacao?

**Regularizacao** e qualquer tecnica que ajuda a rede a generalizar melhor, reduzindo o overfitting. Funciona "dificultando" o treino de forma controlada para que a rede nao decore os dados.

Exemplos neste projeto: Dropout, Weight Decay, Data Augmentation, Early Stopping, Batch Normalization.

---

### O que e Loss (Funcao de Perda)?

A **loss** e uma funcao matematica que mede o quao errada esta a predicao da rede. Quanto menor a loss, mais correta e a predicao.

- Loss = 0: predicao perfeita
- Loss alta: predicao muito errada

O objetivo do treinamento e minimizar a loss. No treino, a loss diminui gradualmente a cada epoca.

---

### Treino vs. Validacao vs. Teste — por que tres conjuntos?

| Conjunto | Tamanho | Para que serve |
|----------|---------|----------------|
| **Treino** | 54.000 | Unico conjunto onde os pesos sao atualizados |
| **Validacao** | 6.000 | Monitora overfitting durante o treino. Nao atualiza pesos. |
| **Teste** | 10.000 | Avaliacao final imparcial. Usado UMA UNICA VEZ ao final. |

Se voce usasse o conjunto de teste para tomar decisoes (escolher arquitetura, parar o treino), estaria "trapaceando" — a rede teria visto indiretamente os dados de teste e as metricas finais seriam infladas.

---

### O que e Hiperparametro?

**Parametros** sao aprendidos pela rede (os pesos). **Hiperparametros** sao configuracoes definidas pelo humano antes do treino. A rede nao aprende hiperparametros — voce os escolhe.

Exemplos: learning rate, batch size, numero de epocas, taxa de dropout, arquitetura da rede.

---

### O que e Logit?

**Logit** e o valor bruto de saida da ultima camada da rede, antes de qualquer normalizacao. Para 10 classes, a rede produz 10 logits — um numero real por classe, sem restricao de escala.

Ex: `[-2.1, 0.3, 8.7, -1.2, ...]` — o maior valor (8.7 na posicao 2) indica que a rede acha que e o digito "2".

Para converter logits em probabilidades, aplica-se o **Softmax**: `softmax(x_i) = exp(x_i) / soma(exp(x_j))`. O resultado soma 1 e pode ser interpretado como confianca.

---

### O que e um Checkpoint?

Um **checkpoint** e um arquivo salvo em disco contendo os pesos da rede em um determinado momento do treinamento. Serve para recuperar o melhor estado da rede sem precisar re-treinar.

---

## PARTE 2 — Estrutura do Projeto e Pipeline

### Como o codigo esta organizado

```
main.py                   → Ponto de entrada. Voce sempre executa este arquivo.
config/
  hyperparameters.yaml    → Configuracoes e hiperparametros padrao
src/
  data_loader.py          → Le os arquivos binarios, cria o Dataset e DataLoaders
  preprocessing.py        → Normalizacao e data augmentation
  architectures.py        → Define as 4 arquiteturas de redes neurais
  training.py             → Loop de treino, otimizadores, schedulers, early stopping
  evaluation.py           → Metricas, graficos, relatorios
  tuning.py               → Busca automatica de hiperparametros (Optuna)
  persistence.py          → Salva, carrega e compara resultados de tuning
results/
  checkpoints/            → Pesos dos melhores modelos (.pt)
  figures/                → Graficos gerados automaticamente
  logs/                   → Arquivo de log do experimento
  best_params.yaml        → Melhor configuracao encontrada pelo tuning
  tuning_history.yaml     → Historico de todos os tunings executados
```

### Os dois modos de execucao

**`python main.py --mode train`**
Treina um modelo unico. Os hiperparametros vem de 3 fontes em ordem de prioridade:
1. Defaults do `config/hyperparameters.yaml` (menor prioridade)
2. `results/best_params.yaml`, se existir (salvo por um tuning anterior)
3. Argumentos da linha de comando como `--architecture DeepCNN` (maior prioridade)

**`python main.py --mode tune --n_trials 50 --sampler tpe`**
Executa 50 tentativas (trials) de treinamento com combinacoes diferentes de hiperparametros. Ao final, salva os melhores e treina o modelo final com eles.

---

## PARTE 3 — Leitura dos Dados (data_loader.py)

### Formato IDX — por que ler bytes diretamente?

O MNIST vem em arquivos `.idx3-ubyte` e `.idx1-ubyte`. Sao arquivos **binarios** — uma sequencia de bytes brutos, sem texto. O projeto os le diretamente, sem usar funcoes prontas do `torchvision`, demonstrando compreensao do formato.

Estrutura do arquivo de imagens:

```
Posicao  Tipo    Valor    Significado
0        int32   2051     Magic number: identifica o tipo de arquivo (0x0803)
4        int32   60000    Numero de imagens
8        int32   28       Altura (pixels)
12       int32   28       Largura (pixels)
16+      uint8   0-255    Pixels das imagens, um byte por pixel
```

- **int32**: inteiro de 32 bits = 4 bytes. Representa numeros grandes como 60.000.
- **big-endian**: o byte mais importante vem primeiro. Diferente de CPUs modernas que usam little-endian, por isso e necessario especificar `>` ao ler.
- **uint8**: unsigned integer de 8 bits = 1 byte. Valores de 0 a 255. Um byte por pixel.

**Leitura em Python:**
```python
magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
```
- `struct.unpack`: desempacota bytes brutos em valores Python.
- `'>IIII'`: 4 inteiros unsigned de 32 bits em big-endian.
- `f.read(16)`: le 16 bytes (4 inteiros × 4 bytes cada).

```python
images = np.frombuffer(raw_data, dtype=np.uint8).reshape(N, 28, 28)
```
- `np.frombuffer`: interpreta os bytes diretamente como array NumPy, sem copias desnecessarias.
- `.reshape(N, 28, 28)`: organiza o array linear de `N*784` bytes em `N` matrizes 28x28.

### Dataset e DataLoader

O PyTorch usa o padrao **Dataset + DataLoader**:

**Dataset** = sabe como pegar uma amostra pelo indice.
```
dataset[0]   → (imagem_tensor, rotulo_0)
dataset[42]  → (imagem_tensor, rotulo_42)
```

**DataLoader** = usa o Dataset para montar batches automaticamente.
```
for imagens, rotulos in dataloader:
    # imagens.shape = (64, 1, 28, 28)
    # rotulos.shape = (64,)
```

Ao criar o Dataset, as imagens sao convertidas:
- `uint8 [0, 255]` → `float32 [0.0, 1.0]` (divide por 255): redes neurais precisam de floats para calcular gradientes.
- Shape `(N, 28, 28)` → `(N, 1, 28, 28)`: adiciona a dimensao de canal. O PyTorch espera imagens no formato **NCHW** (N amostras, C canais, H altura, W largura). MNIST e grayscale, logo C=1.

---

## PARTE 4 — Pre-processamento (preprocessing.py)

### Normalizacao Z-Score

**Formula:** `x_normalizado = (x - media) / desvio_padrao`

Usando os valores do MNIST:
- `media = 0.1307` (calculada sobre todos os pixels de treino)
- `desvio_padrao = 0.3081`

**O que acontece com os pixels apos a normalizacao:**
- Pixel preto (0.0) → `(0.0 - 0.1307) / 0.3081 ≈ -0.42`
- Pixel branco (1.0) → `(1.0 - 0.1307) / 0.3081 ≈ +2.82`

**Por que normalizar?** Sem normalizacao, os gradientes de pixels em escalas diferentes ficam desbalanceados, dificultando o aprendizado. Com normalizacao, a rede aprende mais rapido e de forma mais estavel.

**Regra importante:** a normalizacao do conjunto de teste usa os mesmos valores (0.1307 e 0.3081) calculados no treino. Usar as estatisticas do teste seria "trapaca" — o modelo estaria aproveitando informacoes que nao deveria ter acesso.

### Data Augmentation — aumentar os dados artificialmente

No treino, cada imagem recebe modificacoes aleatorias antes de ser usada:
- **Rotacao aleatoria:** gira a imagem entre -10 e +10 graus
- **Translacao aleatoria:** desloca a imagem ate 10% do seu tamanho (ate ~3 pixels)

**Por que isso ajuda?** A rede ve versoes ligeiramente diferentes da mesma imagem a cada epoca, como se o dataset fosse maior. Isso dificulta que a rede "decore" as imagens exatas, forcando-a a aprender caracteristicas mais gerais.

**Por que nao aplicar no teste?** O teste avalia a rede em dados reais, sem modificacoes. Augmentation e uma tecnica de treinamento, nao de avaliacao.

---

## PARTE 5 — As 4 Arquiteturas (architectures.py)

As arquiteturas seguem uma progressao historica: cada uma introduz uma inovacao sobre a anterior.

---

### Arquitetura 1: MLP (Multilayer Perceptron)

**Notacao:**
```
Input(784) → FC(512) → ReLU → Dropout(0.25) → FC(256) → ReLU → Dropout(0.25) → FC(10)
```

**Lendo a notacao passo a passo:**

**`Input(784)`**
- A imagem 28x28 e "achatada" em um vetor de 784 numeros.
- `nn.Flatten()` faz isso: shape `(B, 1, 28, 28)` → `(B, 784)`.
- Problema: a informacao espacial e perdida. O pixel (5,5) e o pixel (5,6) sao tratados como valores completamente independentes. A rede nao sabe que sao vizinhos.

**`FC(512)`** — Fully Connected (Camada Totalmente Conectada)
- `nn.Linear(784, 512)` no PyTorch.
- Cada um dos **512 neuronios** desta camada recebe todos os 784 valores de entrada.
- Cada neuronio faz: `saida = w₁×entrada₁ + w₂×entrada₂ + ... + w₇₈₄×entrada₇₈₄ + b`
  - onde `w₁...w₇₈₄` sao os pesos e `b` e o bias (deslocamento).
- Numero de parametros: `784 × 512 + 512 = 401.920`
- Shape de saida: `(B, 512)`

**`ReLU`** — Rectified Linear Unit
- Funcao de ativacao aplicada elemento a elemento: `f(x) = max(0, x)`
- Valores positivos passam sem alteracao. Valores negativos viram zero.
- **Por que e necessario?** Sem uma funcao nao-linear entre camadas, empilhar camadas FC seria matematicamente equivalente a uma unica camada FC. A ReLU introduz nao-linearidade, permitindo que a rede aprenda funcoes complexas.
- Saida: `(B, 512)`, com valores em [0, +∞)

**`Dropout(0.25)`**
- Durante o treino: zera aleatoriamente 25% dos 512 valores a cada forward pass.
- Durante a inferencia: desativado. Todos os neuronios sao usados.
- **Por que funciona?** A rede nao pode depender de nenhum neuronio especifico (pois ele pode ser zerado). Isso forca a rede a aprender representacoes redundantes e robustas. Funciona como treinar multiplas redes ao mesmo tempo e combinar suas predicoes.

**`FC(256)`** — `nn.Linear(512, 256)`
- Reduz de 512 para 256.
- Parametros: `512 × 256 + 256 = 131.328`
- O padrao 512 → 256 → 10 e chamado de "funil": a rede e forcada a comprimir a informacao em representacoes cada vez mais abstratas.

**`FC(10)`** — `nn.Linear(256, 10)`
- 10 neuronios de saida = 10 classes (digitos 0-9).
- Parametros: `256 × 10 + 10 = 2.570`
- Produz 10 **logits**: numeros reais sem restricao. Ex: `[-1.2, 0.3, 7.8, -0.5, ...]`
- O maior logit indica a classe predita.

**Total: ~270.000 parametros. Acuracia esperada: ~97.5%**

---

### Arquitetura 2: LeNet-5 (LeCun et al., 1998)

**Notacao:**
```
Input(B,1,28,28)
  → Conv(1→6, 5×5) → ReLU → MaxPool(2×2)     → (B,6,12,12)
  → Conv(6→16, 5×5) → ReLU → MaxPool(2×2)    → (B,16,4,4)
  → Flatten                                    → (B,256)
  → FC(256→120) → ReLU → Dropout
  → FC(120→84)  → ReLU → Dropout
  → FC(84→10)                                  → (B,10)
```

**Inovacao chave: Convolucao**

`Conv(1→6, 5×5)` = `nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)`

Uma **convolucao** funciona deslizando um filtro (matriz de pesos pequena) sobre a imagem inteira:

```
Imagem (28x28)            Filtro (5x5)
[... pixels ...]    ×     [25 pesos]   =   Um Feature Map (24x24)
```

- O filtro desliza por todas as posicoes da imagem (horizontalmente e verticalmente).
- Em cada posicao, calcula o produto interno entre o filtro e o trecho da imagem.
- O resultado e um **feature map** — um mapa indicando onde na imagem esse padrao aparece.

Com 6 filtros, temos 6 feature maps diferentes — cada filtro aprende a detectar um padrao diferente (bordas horizontais, verticais, diagonais, etc.)

**Vantagens sobre o MLP:**
- **Compartilhamento de pesos:** o mesmo filtro 5×5 e usado em todas as posicoes. Em vez de 784 pesos por neuronio, sao apenas 25. O filtro detecta o mesmo padrao onde quer que ele apareca na imagem.
- **Localidade:** cada neuronio so "ve" uma regiao 5×5 da imagem, respeitando a estrutura espacial.

**Calculo do tamanho de saida de uma convolucao:**
```
tamanho_saida = (tamanho_entrada - tamanho_filtro) / stride + 1
              = (28 - 5) / 1 + 1 = 24
```

Portanto: `(B,1,28,28)` → Conv(5×5) → `(B,6,24,24)`

`MaxPool(2×2)` = divide cada feature map em regioes 2×2 e seleciona o maior valor.
- Reduz pela metade: `(B,6,24,24)` → `(B,6,12,12)`
- Sem parametros treinaveis.
- Funcao: reduz o custo computacional e torna a rede menos sensivel a posicao exata de um padrao.

**Concatenando os calculos:**
```
(B,1,28,28) → Conv(5×5) → (B,6,24,24) → Pool(2×2) → (B,6,12,12)
            → Conv(5×5) → (B,16,8,8)  → Pool(2×2) → (B,16,4,4)
            → Flatten → (B,256)   [256 = 16 × 4 × 4]
            → FC(256→120) → FC(120→84) → FC(84→10)
```

**Total: ~44.000 parametros (6x menos que o MLP). Acuracia esperada: ~98.5–99%**

---

### Arquitetura 3: ModernCNN

**Inovacoes sobre a LeNet-5:**

**Batch Normalization (BN)** — `nn.BatchNorm2d(32)`

Normaliza as ativacoes de cada camada usando a media e variancia do mini-batch atual. Apos a normalizacao, aplica dois parametros treinaveis: escala (γ) e deslocamento (β).

Problema que resolve: conforme os pesos mudam durante o treino, a distribuicao dos valores que chegam em cada camada muda tambem. Isso e o "Internal Covariate Shift" — a camada precisa ficar se readaptando. BN estabiliza essa distribuicao, permitindo treinar mais rapido com learning rates maiores.

**Convolucoes 3×3 empilhadas em vez de uma 5×5**

Duas convolucoess 3×3 seguidas tem o mesmo "campo de visao" que uma 5×5, mas:
- Menos parametros: `2 × (3×3) = 18` vs. `5×5 = 25`
- Mais nao-linearidades (duas ReLUs vs. uma)

**Dropout2d (Spatial Dropout)** — `nn.Dropout2d(p=0.125)`

Versao do Dropout para feature maps: em vez de zerar neuronios individuais, zera **feature maps inteiros**. Mais eficaz para dados espaciais porque pixels vizinhos sao correlacionados — desativar um neuronio individual tem pouco efeito pois os vizinhos fornecem a mesma informacao.

**Notacao:**
```
(B,1,28,28)
  → Conv(1→32,3×3) → BN → ReLU → Conv(32→32,3×3) → BN → ReLU → Pool → Dropout2d
  → Conv(32→64,3×3) → BN → ReLU → Conv(64→64,3×3) → BN → ReLU → Pool → Dropout2d
  → Flatten(B,3136)
  → FC(3136→256) → BN1d → ReLU → Dropout
  → FC(256→10)
```

**Total: ~160.000 parametros. Acuracia esperada: ~99%**

---

### Arquitetura 4: DeepCNN (Blocos Residuais)

**O problema que resolve: Vanishing Gradient**

Em redes muito profundas, o gradiente que volta pelo backpropagation fica cada vez menor a cada camada que percorre. Nas camadas iniciais, o gradiente chega tao pequeno que elas praticamente nao aprendem. Isso e o **vanishing gradient** (gradiente que some).

**A solucao: Skip Connection (conexao de atalho)**

```
Sem residual:   entrada → [camadas] → saida
Com residual:   entrada → [camadas] → soma com a entrada original → saida
```

Em formula: `saida = F(entrada) + entrada`

A soma com a entrada original cria um "caminho curto" pelo qual o gradiente pode fluir diretamente, sem passar pelas camadas intermediarias. Mesmo que as camadas intermediarias tenham gradientes pequenos, o gradiente chega pelas conexoes de atalho.

**Notacao do ResidualBlock:**
```
entrada (x)
  ├──→ Conv(3×3) → BN → ReLU → Conv(3×3) → BN → F(x)
  │
  └──────────────────────────────────────────→ x
                                               ↓
                                          F(x) + x → ReLU → saida
```

**Notacao completa da DeepCNN:**
```
(B,1,28,28)
  [Stem]
  → Conv(1→32,3×3) → BN → ReLU                   → (B,32,28,28)

  [Stage 1 — 2 blocos residuais]
  → ResBlock(32) → ResBlock(32) → MaxPool → Dropout2d  → (B,32,14,14)

  [Transicao — muda o numero de canais]
  → Conv(32→64, 1×1) → BN → ReLU                 → (B,64,14,14)

  [Stage 2 — 2 blocos residuais]
  → ResBlock(64) → ResBlock(64) → MaxPool → Dropout2d  → (B,64,7,7)

  [Classificador]
  → AdaptiveAvgPool2d(1)                           → (B,64,1,1)
  → Flatten                                        → (B,64)
  → FC(64→128) → BN1d → ReLU → Dropout
  → FC(128→10)                                     → (B,10)
```

**`Conv(32→64, 1×1)`** — Convolucao 1×1 (Pointwise Convolution):
Um filtro de tamanho 1×1 opera em cada posicao espacial individualmente, combinando os 32 canais em 64. Nao muda as dimensoes espaciais (H, W). E a forma de mudar o numero de canais sem alterar a resolucao.

**`AdaptiveAvgPool2d(1)`** — Global Average Pooling:
Calcula a media de cada feature map inteiro, reduzindo qualquer tamanho para 1×1. `(B,64,7,7)` → `(B,64,1,1)`. Comprime toda a informacao espacial de cada canal em um unico numero, funcionando como um classificador implicito forte.

**Total: ~125.000 parametros. Acuracia esperada: ~99–99.5%**

---

## PARTE 6 — Treinamento (training.py)

### O ciclo de aprendizado — o que acontece a cada batch

Para cada grupo de 64 imagens:

**Passo 1 — Zeramos os gradientes acumulados**
```python
optimizer.zero_grad()
```
O PyTorch acumula gradientes por padrao (soma). Se nao zerarmos antes de cada batch, os gradientes do batch atual seriam somados aos do anterior — errado.

**Passo 2 — Forward Pass: calculamos a predicao**
```python
outputs = model(images)   # shape: (64, 10) — 10 logits por imagem
```
As imagens passam por todas as camadas em sequencia ate produzir os logits.

**Passo 3 — Calculamos o erro (Loss)**
```python
loss = criterion(outputs, labels)  # CrossEntropyLoss
```
`CrossEntropyLoss` mede o quao longe os logits estao dos rotulos corretos. Combina duas operacoes:
1. **Softmax**: converte os 10 logits em 10 probabilidades que somam 1.
2. **Negative Log-Likelihood**: penaliza a log-probabilidade da classe correta.

Resultado: um numero escalar. Ex: `loss = 0.23`.

**Passo 4 — Backward Pass: calculamos os gradientes**
```python
loss.backward()
```
Percorre toda a rede de tras para frente e calcula, para cada parametro, o quanto ele contribuiu para o erro. Esse valor e o **gradiente** — armazenado em `param.grad`.

**Passo 5 — Atualizamos os pesos**
```python
optimizer.step()
```
Usa os gradientes para ajustar cada parametro na direcao que reduz o erro.

---

### Otimizadores — como os pesos sao atualizados

**SGD (Stochastic Gradient Descent)**
```
novo_peso = peso_atual - learning_rate × gradiente
```
O mais simples: da um passo na direcao oposta ao gradiente.

Com **Momentum** (adiciona "inercia"):
```
velocidade = 0.9 × velocidade_anterior + gradiente
novo_peso  = peso_atual - learning_rate × velocidade
```
O momentum acumula o historico de gradientes como uma "velocidade". Em direcoes consistentes, a velocidade cresce (aceleracao). Em direcoes que oscilam, a velocidade se cancela (amortecimento).

Com **Nesterov**: antes de calcular o gradiente, "antecipa" onde o momentum levaria e calcula o gradiente la. Convergencia mais rapida.

**Adam (Kingma & Ba, 2015)**

Adam adapta a learning rate para cada parametro individualmente:
- Parametros que recebem gradientes grandes e consistentes: learning rate efetiva menor.
- Parametros que recebem gradientes pequenos ou esparsos: learning rate efetiva maior.

Usa duas medias moveis do gradiente: a media simples (1o momento) e a media dos quadrados (2o momento). A atualizacao e proporcional a `media / sqrt(variancia)`.

**AdamW**
O Adam original tem um problema: a regularizacao L2 (weight decay) interage mal com a adaptacao de learning rate. O AdamW corrige isso aplicando o weight decay diretamente nos pesos, separado da atualizacao adaptativa. Resulta em melhor generalizacao.

---

### Schedulers — como a learning rate muda ao longo do treino

**Por que mudar a lr?** No inicio, queremos explorar rapido (lr alta). No final, queremos convergir com precisao (lr baixa).

**StepLR**: divide a lr por 10 a cada terco do treino. Ex: lr=0.01 → 0.001 → 0.0001.

**CosineAnnealingLR**: a lr segue uma curva cosseno, caindo suavemente do valor inicial ate proximo de zero.

**ReduceLROnPlateau**: monitora a val_loss. Se ela nao melhorar por 5 epocas consecutivas, divide a lr pela metade. Reativo — so atua quando necessario.

---

### Early Stopping — parar na hora certa

O treinamento monitora a `val_loss` a cada epoca. Se ela nao melhorar por `patience=7` epocas consecutivas, o treino para.

O modelo salvo e o dos **melhores pesos ja vistos** (melhor val_loss), nao o da ultima epoca. Isso evita que o modelo retorne com overfitting.

---

## PARTE 7 — Avaliacao (evaluation.py)

### Acuracia (Accuracy)
```
acuracia = predicoes_corretas / total_de_exemplos
```
A metrica mais intuitiva. Um valor de 0.9923 significa que 99.23% das predicoes estao corretas.

### Precision, Recall e F1 por classe

Para o digito "5", por exemplo:
- **Precision**: de todas as vezes que a rede disse "e um 5", em quantas estava certa?
- **Recall**: de todas as imagens reais de 5 no dataset, quantas a rede identificou corretamente?
- **F1**: media harmonica de Precision e Recall. Penaliza quando um dos dois e muito baixo.

### Matriz de Confusao

Tabela 10×10 mostrando quais digitos sao confundidos entre si. A diagonal sao os acertos. Elementos fora da diagonal sao erros sistematicos. Ex: se a celula `[4][9]` for alta, a rede esta confundindo muitos "4" com "9".

### Curvas de Aprendizado

Grafico de loss e accuracy por epoca, treino vs. validacao. Serve para diagnosticar overfitting (treino melhorando mas validacao estagnando) ou underfitting (ambas ruins).

---

## PARTE 8 — Otimizacao de Hiperparametros (tuning.py)

### O problema

Quais valores usar para learning rate, arquitetura, otimizador, etc.? Testar todas as combinacoes manualmente seria inviavel. O tuning automatiza essa busca.

### Como o Optuna funciona

O Optuna chama uma funcao (chamada funcao objetivo) N vezes. Cada chamada:
1. O Optuna **sugere** valores para os 8 hiperparametros.
2. Um modelo e **treinado** com esses valores.
3. A **acuracia de validacao** e retornada.
4. O Optuna usa esse resultado para guiar as proximas sugestoes.

### As tres estrategias de busca

**Random Search** — busca aleatoria
Sorteia combinacoes aleatorias. Simples, mas funciona bem porque cada trial explora uma combinacao unica de todos os hiperparametros.

**TPE (Tree-structured Parzen Estimator)** — busca bayesiana
Aprende com os trials anteriores. Identifica quais regioes do espaco de hiperparametros tendem a dar bons resultados e concentra a busca nessas regioes. Mais eficiente que Random Search, especialmente com muitos trials.

**Grid Search** — busca exhaustiva
Testa todas as combinacoes de um grid fixo. Garante cobertura total mas e muito caro para espacos grandes.

### Pruning — economia de recursos

O Optuna pode interromper um trial cedo se ele estiver indo muito mal em relacao aos outros. O `MedianPruner` compara a acuracia de cada trial com a mediana dos trials anteriores na mesma epoca. Se estiver muito abaixo, o trial e cancelado — economizando o tempo que seria gasto nas epocas restantes.

---

## PARTE 9 — Sistema de Persistencia (persistence.py)

### O problema que resolve

Sem persistencia:
1. Voce roda o tuning, encontra os melhores hiperparametros.
2. Para treinar o modelo final, voce teria que anotar os valores e digitar na linha de comando.
3. Se voce rodar um segundo tuning, nao saberia automaticamente se foi melhor ou pior que o primeiro.

O sistema de persistencia automatiza tudo isso.

### Como funciona

**Apos o tuning:**
Os melhores hiperparametros sao salvos em `results/best_params.yaml`.

**Ao rodar `--mode train`:**
O pipeline detecta automaticamente o `best_params.yaml` e usa esses hiperparametros, sem precisar digitar nada.

**Ao rodar um segundo tuning:**
O novo resultado e comparado com o salvo. O melhor permanece como o best. Uma tabela de comparacao e exibida no log.

### Criterios de comparacao entre dois tunings

Quando dois tunings produzem resultados similares, como decidir qual e melhor? O sistema usa uma hierarquia:

1. **Acuracia de validacao** (principal): maior e melhor. Diferenca menor que 0.01% e tratada como empate.
2. **Numero de parametros** (desempate): menor e melhor — modelos mais simples generalizam melhor e sao mais rapidos.
3. **Tempo de treinamento** (desempate final): menor e melhor.

### Tres niveis de prioridade para os hiperparametros

Ao rodar `--mode train`, os hiperparametros sao resolvidos assim:

```
1. Carrega os defaults do config/hyperparameters.yaml
2. Se best_params.yaml existir, seus valores sobreescrevem os defaults
3. Se voce passar --architecture DeepCNN na linha de comando, isso sobreescreve tudo
```

A flag `--no_saved_params` pula o passo 2, usando apenas YAML e linha de comando.

---

## PARTE 10 — Reproducibilidade

### Por que os resultados precisam ser reproduziveis?

Na ciencia, um resultado so e valido se outra pessoa (ou voce mesmo em outro momento) consegue reproduzi-lo exatamente. Em ML, existem varias fontes de aleatoriedade que precisam ser controladas.

### Fixacao de sementes

Uma **semente** (seed) inicializa um gerador de numeros aleatorios de forma deterministica. Com a mesma semente, o mesmo gerador sempre produz a mesma sequencia de numeros aleatorios.

O codigo fixa a semente em todos os lugares onde ha aleatoriedade:

```python
random.seed(42)              # Aleatoriedade do Python (augmentation)
np.random.seed(42)           # Aleatoriedade do NumPy
torch.manual_seed(42)        # Inicializacao dos pesos e shuffling (CPU)
torch.cuda.manual_seed_all(42)  # GPU
torch.backends.cudnn.deterministic = True  # Algoritmos deterministicos na GPU
torch.backends.cudnn.benchmark = False     # Desabilita selecao automatica de algoritmos
```

Com todas essas sementes fixadas, duas execucoes com os mesmos dados e hiperparametros produzem exatamente os mesmos resultados.

**Trade-off:** `cudnn.deterministic=True` pode ser 10-20% mais lento, pois a GPU nao pode escolher o algoritmo mais rapido (que pode variar entre execucoes).

---

## Resumo dos Comandos

```bash
# Treino simples (usa defaults do YAML ou best_params se existir)
python main.py --mode train

# Treino com hiperparametros especificos
python main.py --mode train --architecture DeepCNN --epochs 30

# Tuning com 50 trials (busca bayesiana)
python main.py --mode tune --n_trials 50 --sampler tpe

# Tuning com busca aleatoria
python main.py --mode tune --n_trials 30 --sampler random

# Treino ignorando resultados de tuning anteriores
python main.py --mode train --no_saved_params
```

---

## Referencias Bibliograficas

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278–2324.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. arXiv:1512.03385.
3. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." *ICML*.
4. Srivastava, N. et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15, 1929–1958.
5. Kingma, D. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
6. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
7. Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*.
8. Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD*.
9. Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR*, 13, 281–305.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.
11. Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.
12. Simard, P., Steinkraus, D., & Platt, J. (2003). "Best Practices for CNNs Applied to Visual Document Analysis." *ICDAR*.
13. Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533–536.
14. Simonyan, K. & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
15. Prechelt, L. (1998). "Early Stopping — But When?" In *Neural Networks: Tricks of the Trade*.
16. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning." Springer.
