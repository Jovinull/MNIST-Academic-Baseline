# ============================================================================
# MNIST Academic Baseline — Módulo de Carregamento de Dados
# ============================================================================
# Este módulo implementa a leitura direta dos arquivos binários IDX do MNIST,
# sem depender de funções prontas como torchvision.datasets.MNIST. Isso
# demonstra compreensão do formato de armazenamento e das operações de I/O
# de baixo nível necessárias para trabalhar com dados em cenários reais.
#
# FORMATO IDX (LeCun et al., 1998):
# O formato IDX é um formato binário simples para vetores e matrizes
# multidimensionais de vários tipos numéricos.
#
# Estrutura do arquivo de IMAGENS (idx3-ubyte):
#   [offset] [tipo]   [valor]          [descrição]
#   0000     int32    0x00000803       magic number (2051)
#   0004     int32    N                número de imagens
#   0008     int32    28               número de linhas (pixels)
#   0012     int32    28               número de colunas (pixels)
#   0016     uint8    pixel            dados dos pixels [0, 255]
#   ...      uint8    pixel            (N × 28 × 28 bytes no total)
#
# Estrutura do arquivo de RÓTULOS (idx1-ubyte):
#   [offset] [tipo]   [valor]          [descrição]
#   0000     int32    0x00000801       magic number (2049)
#   0004     int32    N                número de rótulos
#   0008     uint8    label            classe do dígito [0, 9]
#   ...      uint8    label            (N bytes no total)
#
# Referências:
#   - LeCun, Y., Cortes, C., & Burges, C.J. (1998). "The MNIST Database
#     of Handwritten Digits." http://yann.lecun.com/exdb/mnist/
#   - Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning",
#     MIT Press. Capítulo 1 — representação de dados.
# ============================================================================

import struct       # Módulo padrão para desempacotar dados binários
import numpy as np  # Computação numérica eficiente (Harris et al., 2020)
import os           # Manipulação de caminhos de arquivo
import logging      # Registro estruturado de eventos do pipeline

import torch                          # Framework de tensores (Paszke et al., 2019)
from torch.utils.data import (
    Dataset,                          # Classe base para datasets customizados
    DataLoader,                       # Iterador com batching e shuffling
    random_split,                     # Particionamento aleatório de datasets
)

# Configura o logger deste módulo para rastrear operações de I/O
logger = logging.getLogger(__name__)


# ============================================================================
# FUNÇÕES DE PARSING DOS ARQUIVOS BINÁRIOS IDX
# ============================================================================

def parse_idx_images(filepath: str) -> np.ndarray:
    """
    Lê um arquivo binário IDX3 contendo imagens e retorna um array NumPy.

    O parsing segue rigorosamente a especificação do formato IDX:
    1. Abre o arquivo em modo binário ('rb' — read binary).
    2. Lê os primeiros 16 bytes do cabeçalho (header).
    3. Desempacota o magic number e as dimensões usando struct.unpack.
    4. Lê o restante do arquivo como um buffer de bytes contíguos.
    5. Converte para um array NumPy e reshape para (N, 28, 28).

    Parâmetros:
        filepath (str): Caminho absoluto ou relativo ao arquivo .idx3-ubyte.

    Retorna:
        np.ndarray: Array de shape (N, 28, 28) com dtype uint8, onde N é o
                    número de imagens e cada pixel está no intervalo [0, 255].

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho especificado.
        ValueError: Se o magic number não corresponder ao esperado (2051).
    """
    # Verifica existência do arquivo antes de tentar abri-lo.
    # Isso produz mensagens de erro mais claras do que deixar o OS levantar
    # uma exceção genérica.
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Arquivo de imagens não encontrado: {filepath}\n"
            f"Baixe os arquivos do MNIST em http://yann.lecun.com/exdb/mnist/"
        )

    # Abre o arquivo em modo leitura binária.
    # O context manager 'with' garante que o arquivo será fechado mesmo
    # se ocorrer uma exceção durante a leitura.
    with open(filepath, 'rb') as f:
        # -----------------------------------------------------------------
        # LEITURA DO CABEÇALHO (16 bytes)
        # -----------------------------------------------------------------
        # '>IIII' especifica o formato de desempacotamento:
        #   '>' = big-endian (byte order do formato IDX)
        #   'I'  = unsigned int de 32 bits (4 bytes cada)
        #   4 valores × 4 bytes = 16 bytes totais do cabeçalho
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))

        # Validação do magic number para garantir integridade do arquivo.
        # O valor 2051 (0x0803) indica um arquivo IDX3 de unsigned bytes.
        if magic != 2051:
            raise ValueError(
                f"Magic number inválido: {magic}. Esperado: 2051. "
                f"O arquivo pode estar corrompido ou não ser um IDX3."
            )

        # Registra informações sobre os dados carregados para diagnóstico.
        logger.info(
            f"Arquivo de imagens carregado: {num_images} imagens de "
            f"{num_rows}×{num_cols} pixels"
        )

        # -----------------------------------------------------------------
        # LEITURA DOS PIXELS
        # -----------------------------------------------------------------
        # Após o cabeçalho de 16 bytes, o restante do arquivo contém os
        # valores dos pixels como bytes unsigned (uint8, intervalo [0, 255]).
        #
        # np.frombuffer interpreta o buffer de bytes diretamente como um
        # array NumPy, evitando cópias desnecessárias em memória.
        # Isso é mais eficiente do que ler byte a byte em um loop Python.
        raw_data = f.read()
        images = np.frombuffer(raw_data, dtype=np.uint8)

        # Reshape de um vetor 1D para um tensor 3D: (N, H, W)
        # onde N = número de imagens, H = altura, W = largura.
        images = images.reshape(num_images, num_rows, num_cols)

    return images


def parse_idx_labels(filepath: str) -> np.ndarray:
    """
    Lê um arquivo binário IDX1 contendo rótulos e retorna um array NumPy.

    Segue a mesma lógica de parsing do arquivo de imagens, mas com um
    cabeçalho menor (8 bytes) e dados unidimensionais.

    Parâmetros:
        filepath (str): Caminho absoluto ou relativo ao arquivo .idx1-ubyte.

    Retorna:
        np.ndarray: Array de shape (N,) com dtype uint8, onde cada elemento
                    é um inteiro no intervalo [0, 9] representando a classe.

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho especificado.
        ValueError: Se o magic number não corresponder ao esperado (2049).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Arquivo de rótulos não encontrado: {filepath}\n"
            f"Baixe os arquivos do MNIST em http://yann.lecun.com/exdb/mnist/"
        )

    with open(filepath, 'rb') as f:
        # -----------------------------------------------------------------
        # CABEÇALHO DO ARQUIVO DE RÓTULOS (8 bytes)
        # -----------------------------------------------------------------
        # '>II' = 2 × unsigned int 32-bit em big-endian.
        # O magic number 2049 (0x0801) identifica um arquivo IDX1.
        magic, num_labels = struct.unpack('>II', f.read(8))

        if magic != 2049:
            raise ValueError(
                f"Magic number inválido: {magic}. Esperado: 2049. "
                f"O arquivo pode estar corrompido ou não ser um IDX1."
            )

        logger.info(f"Arquivo de rótulos carregado: {num_labels} rótulos")

        # -----------------------------------------------------------------
        # LEITURA DOS RÓTULOS
        # -----------------------------------------------------------------
        # Cada rótulo ocupa 1 byte (uint8). O array resultante tem shape (N,).
        raw_data = f.read()
        labels = np.frombuffer(raw_data, dtype=np.uint8)

    return labels


# ============================================================================
# DATASET CUSTOMIZADO PARA PYTORCH
# ============================================================================

class MNISTDataset(Dataset):
    """
    Dataset customizado que encapsula os arrays NumPy do MNIST em uma
    interface compatível com o DataLoader do PyTorch.

    A classe torch.utils.data.Dataset exige a implementação de dois métodos:
      - __len__:     retorna o número total de amostras.
      - __getitem__: retorna a i-ésima amostra (imagem, rótulo).

    Este padrão é fundamental no PyTorch, pois o DataLoader depende desses
    métodos para criar batches, embaralhar dados e paralelizar o carregamento
    (Paszke et al., 2019, seção 3 — "Data Loading and Processing").

    Atributos:
        images (torch.Tensor): Tensor de imagens com shape (N, 1, 28, 28),
                               normalizado para [0, 1] e em float32.
        labels (torch.Tensor): Tensor de rótulos com shape (N,) em int64.
        transform (callable):  Transformação opcional aplicada a cada imagem.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Inicializa o dataset convertendo arrays NumPy em tensores PyTorch.

        Parâmetros:
            images (np.ndarray): Array de shape (N, 28, 28), dtype uint8.
            labels (np.ndarray): Array de shape (N,), dtype uint8.
            transform (callable, optional): Função de transformação aplicada
                                            a cada imagem no __getitem__.
        """
        # -----------------------------------------------------------------
        # CONVERSÃO E NORMALIZAÇÃO DAS IMAGENS
        # -----------------------------------------------------------------
        # 1. Converte uint8 [0, 255] → float32 [0.0, 1.0] dividindo por 255.
        #    Redes neurais convergem melhor com entradas normalizadas, pois
        #    os gradientes ficam em escalas comparáveis para todos os pixels
        #    (LeCun et al., 1998; Goodfellow et al., 2016, cap. 8).
        #
        # 2. Adiciona dimensão de canal: (N, 28, 28) → (N, 1, 28, 28).
        #    O PyTorch espera tensores no formato NCHW (batch, canais, altura,
        #    largura), que é o layout padrão para convoluções 2D.
        #    MNIST é grayscale, então C=1.
        self.images = torch.tensor(
            images, dtype=torch.float32
        ).unsqueeze(1) / 255.0

        # Converte rótulos para int64 (LongTensor), tipo requerido pela
        # função de perda CrossEntropyLoss do PyTorch.
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Armazena a transformação opcional (e.g., data augmentation).
        self.transform = transform

    def __len__(self) -> int:
        """
        Retorna o número total de amostras no dataset.

        O DataLoader chama este método para calcular o número de batches
        por época: num_batches = ceil(len(dataset) / batch_size).
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Retorna a amostra no índice 'idx' como uma tupla (imagem, rótulo).

        Este método é chamado pelo DataLoader para cada amostra individual.
        Se uma transformação foi definida, ela é aplicada à imagem antes
        de retorná-la. Isso permite aplicar data augmentation de forma
        "lazy" (sob demanda), economizando memória.

        Parâmetros:
            idx (int): Índice da amostra desejada, no intervalo [0, N-1].

        Retorna:
            tuple: (image, label) onde image é um tensor (1, 28, 28)
                   e label é um escalar int64.
        """
        # Seleciona a imagem e o rótulo pelo índice.
        image = self.images[idx]
        label = self.labels[idx]

        # Aplica transformação, se definida.
        # Transformações típicas incluem rotação, translação e normalização
        # adicional (ver módulo preprocessing.py).
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================================
# FUNÇÃO PRINCIPAL DE CARREGAMENTO E PARTICIONAMENTO
# ============================================================================

def load_mnist(
    data_dir: str,
    batch_size: int = 64,
    validation_split: float = 0.1,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    train_transform=None,
    test_transform=None,
):
    """
    Função de alto nível que orquestra todo o pipeline de carregamento:
    1. Lê os 4 arquivos binários IDX do MNIST.
    2. Cria datasets PyTorch com conversão e normalização.
    3. Particiona o treino em treino + validação.
    4. Retorna DataLoaders prontos para consumo pelo loop de treinamento.

    A separação em treino/validação/teste é um princípio fundamental em
    Machine Learning (Hastie et al., 2009, "The Elements of Statistical
    Learning", cap. 7). O conjunto de validação permite:
      - Monitorar overfitting durante o treinamento.
      - Selecionar hiperparâmetros sem contaminar o conjunto de teste.
      - Implementar early stopping baseado em métricas de validação.

    Parâmetros:
        data_dir (str): Diretório contendo os 4 arquivos .idx*-ubyte.
        batch_size (int): Tamanho do mini-batch. Valores típicos: 32-256.
        validation_split (float): Fração do treino para validação (0.0-1.0).
        num_workers (int): Processos paralelos para carregamento de dados.
        pin_memory (bool): Se True, usa memória paginada para GPU.
        seed (int): Semente para reprodutibilidade do split.
        train_transform (callable): Transformação para dados de treino.
        test_transform (callable): Transformação para dados de teste.

    Retorna:
        dict: Dicionário com chaves 'train', 'val' e 'test', cada uma
              contendo um DataLoader configurado apropriadamente.
    """
    # -----------------------------------------------------------------
    # NOMES DOS ARQUIVOS BINÁRIOS IDX
    # -----------------------------------------------------------------
    # O MNIST é distribuído em 4 arquivos separados:
    #   - Imagens de treino: 60.000 exemplos (train-images.idx3-ubyte)
    #   - Rótulos de treino: 60.000 rótulos (train-labels.idx1-ubyte)
    #   - Imagens de teste: 10.000 exemplos  (t10k-images.idx3-ubyte)
    #   - Rótulos de teste: 10.000 rótulos   (t10k-labels.idx1-ubyte)
    file_names = {
        'train_images': 'train-images.idx3-ubyte',
        'train_labels': 'train-labels.idx1-ubyte',
        'test_images':  't10k-images.idx3-ubyte',
        'test_labels':  't10k-labels.idx1-ubyte',
    }

    # Constrói os caminhos completos concatenando o diretório base.
    paths = {
        key: os.path.join(data_dir, filename)
        for key, filename in file_names.items()
    }

    # -----------------------------------------------------------------
    # PARSING DOS ARQUIVOS BINÁRIOS
    # -----------------------------------------------------------------
    logger.info("Iniciando parsing dos arquivos binários IDX...")

    train_images = parse_idx_images(paths['train_images'])
    train_labels = parse_idx_labels(paths['train_labels'])
    test_images  = parse_idx_images(paths['test_images'])
    test_labels  = parse_idx_labels(paths['test_labels'])

    # Validação de consistência: o número de imagens deve ser igual ao
    # número de rótulos para cada split.
    assert len(train_images) == len(train_labels), (
        f"Inconsistência nos dados de treino: {len(train_images)} imagens "
        f"vs {len(train_labels)} rótulos."
    )
    assert len(test_images) == len(test_labels), (
        f"Inconsistência nos dados de teste: {len(test_images)} imagens "
        f"vs {len(test_labels)} rótulos."
    )

    logger.info(
        f"Dados carregados — Treino: {len(train_images)}, "
        f"Teste: {len(test_images)}"
    )

    # -----------------------------------------------------------------
    # CRIAÇÃO DOS DATASETS PYTORCH
    # -----------------------------------------------------------------
    full_train_dataset = MNISTDataset(
        train_images, train_labels, transform=train_transform
    )
    test_dataset = MNISTDataset(
        test_images, test_labels, transform=test_transform
    )

    # -----------------------------------------------------------------
    # PARTICIONAMENTO TREINO → TREINO + VALIDAÇÃO
    # -----------------------------------------------------------------
    # Calcula o tamanho de cada partição.
    total_train = len(full_train_dataset)
    val_size    = int(total_train * validation_split)
    train_size  = total_train - val_size

    # Usa um Generator com semente fixa para garantir que o split seja
    # reprodutível entre execuções diferentes.
    # Referência: PyTorch Docs — torch.utils.data.random_split.
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    logger.info(
        f"Split realizado — Treino: {train_size}, Validação: {val_size}, "
        f"Teste: {len(test_dataset)}"
    )

    # -----------------------------------------------------------------
    # CRIAÇÃO DOS DATALOADERS
    # -----------------------------------------------------------------
    # DataLoaders encapsulam a lógica de iteração sobre o dataset:
    #   - Agrupamento em mini-batches (batch_size).
    #   - Embaralhamento aleatório (shuffle) — apenas no treino.
    #   - Carregamento paralelo (num_workers) para throughput.
    #   - Pin memory para transferência eficiente CPU→GPU.
    #
    # IMPORTANTE: shuffle=True apenas no treino. A apresentação aleatória
    # dos exemplos a cada época reduz a correlação entre batches
    # consecutivos, melhorando a convergência do SGD
    # (Bottou, 2012 — "Stochastic Gradient Descent Tricks").
    #
    # Validação e teste usam shuffle=False para garantir resultados
    # determinísticos e comparáveis entre execuções.

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,             # Embaralha a cada época
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,           # Mantém o último batch mesmo se incompleto
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,             # Sem embaralhamento na validação
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,             # Sem embaralhamento no teste
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return dataloaders
