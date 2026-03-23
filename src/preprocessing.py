# ============================================================================
# MNIST Academic Baseline — Módulo de Pré-processamento
# ============================================================================
# Este módulo implementa as transformações aplicadas aos dados antes e
# durante o treinamento. O pré-processamento é uma etapa crítica no pipeline
# de Machine Learning, pois dados mal condicionados podem impedir a
# convergência ou degradar severamente o desempenho do modelo.
#
# As transformações implementadas aqui seguem duas categorias:
#
# 1. NORMALIZAÇÃO ESTATÍSTICA:
#    Centraliza os dados em torno de média zero e variância unitária.
#    Fundamentação: LeCun et al. (1998, "Efficient BackProp") demonstram que
#    entradas normalizadas aceleram a convergência, pois as superfícies de
#    erro se tornam mais esféricas, facilitando a navegação pelo gradiente.
#
# 2. DATA AUGMENTATION (aumento de dados):
#    Gera variações artificiais dos exemplos de treino para aumentar a
#    robustez do modelo e reduzir overfitting. Essa técnica é uma forma
#    implícita de regularização.
#    Fundamentação: Simard et al. (2003, "Best Practices for Convolutional
#    Neural Networks Applied to Visual Document Analysis") mostram que
#    distorções elásticas e afins em dígitos manuscritos reduzem o erro
#    em até 40% em relação ao treinamento sem augmentation.
#
# Referências adicionais:
#   - Shorten & Khoshgoftaar (2019), "A survey on Image Data Augmentation
#     for Deep Learning", Journal of Big Data.
#   - Goodfellow et al. (2016), "Deep Learning", cap. 7.4 — Dataset
#     Augmentation como regularização.
# ============================================================================

import torch                       # Operações com tensores
import torchvision.transforms as T # Transformações compostas do PyTorch

# ============================================================================
# CONSTANTES DE NORMALIZAÇÃO DO MNIST
# ============================================================================
# Média e desvio padrão calculados sobre todo o conjunto de treino do MNIST.
# Esses valores são amplamente reportados na literatura e utilizados como
# padrão em benchmarks (ver repositório oficial do PyTorch Examples).
#
# Média ≈ 0.1307: o MNIST tem fundo preto (0) dominante e traços brancos
# (próximos de 1), resultando em uma média baixa. Isso reflete o fato de
# que a maioria dos pixels em uma imagem 28×28 pertence ao fundo.
#
# Desvio padrão ≈ 0.3081: a variância é moderada, pois há uma mistura
# bimodal entre pixels de fundo (≈0) e pixels de traço (≈1).
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


def get_train_transforms(normalize: bool = True, augment: bool = True):
    """
    Constrói o pipeline de transformações para o conjunto de TREINO.

    No treinamento, aplicamos duas etapas sequenciais:
    1. Data augmentation (se habilitada) — introduz variabilidade.
    2. Normalização estatística (se habilitada) — padroniza a escala.

    A composição é feita via torchvision.transforms.Compose, que aplica
    as transformações na ordem em que são listadas (pipeline sequencial).

    Parâmetros:
        normalize (bool): Se True, aplica normalização z-score.
        augment (bool): Se True, aplica transformações de augmentation.

    Retorna:
        torchvision.transforms.Compose: Pipeline de transformações.
    """
    # Lista que acumulará as transformações a serem compostas.
    transforms_list = []

    if augment:
        # -----------------------------------------------------------------
        # ROTAÇÃO ALEATÓRIA (±10 graus)
        # -----------------------------------------------------------------
        # Dígitos manuscritos naturalmente apresentam variação na inclinação.
        # Uma rotação de ±10° simula essa variação sem distorcer o dígito
        # a ponto de torná-lo irreconhecível.
        # Valor escolhido com base em Simard et al. (2003), que recomendam
        # perturbações suaves para o MNIST.
        transforms_list.append(T.RandomRotation(degrees=10))

        # -----------------------------------------------------------------
        # TRANSLAÇÃO ALEATÓRIA (±10% em cada eixo)
        # -----------------------------------------------------------------
        # Simula variação na posição do dígito dentro da imagem.
        # O MNIST original centraliza os dígitos, mas em dados reais a
        # posição pode variar. Translate=(0.1, 0.1) permite deslocamentos
        # de até 2-3 pixels em cada direção (10% de 28 ≈ 2.8 pixels).
        transforms_list.append(
            T.RandomAffine(
                degrees=0,                  # Sem rotação adicional
                translate=(0.1, 0.1),        # Translação horizontal e vertical
            )
        )

    if normalize:
        # -----------------------------------------------------------------
        # NORMALIZAÇÃO Z-SCORE
        # -----------------------------------------------------------------
        # Transforma os pixels de [0, 1] para distribuição com:
        #   média ≈ 0, desvio padrão ≈ 1
        #
        # Fórmula aplicada pixel a pixel: x_norm = (x - mean) / std
        #
        # Por que normalizar?
        #   1. Gradientes ficam em escalas comparáveis → convergência
        #      mais rápida e estável (LeCun et al., 1998).
        #   2. Evita saturação em funções de ativação como sigmoid/tanh,
        #      mantendo as ativações na região de maior gradiente.
        #   3. Facilita o uso de taxas de aprendizado maiores, pois a
        #      paisagem de perda se torna mais suave.
        transforms_list.append(T.Normalize(mean=MNIST_MEAN, std=MNIST_STD))

    # Se nenhuma transformação foi adicionada, retorna a identidade.
    if not transforms_list:
        return None

    # Compose encadeia as transformações em um pipeline sequencial.
    return T.Compose(transforms_list)


def get_test_transforms(normalize: bool = True):
    """
    Constrói o pipeline de transformações para os conjuntos de VALIDAÇÃO
    e TESTE.

    IMPORTANTE: No teste/validação, NÃO aplicamos data augmentation.
    A avaliação deve ser determinística e refletir o desempenho real do
    modelo sobre dados "limpos". Augmentation é uma técnica de treinamento,
    não de avaliação (Goodfellow et al., 2016, cap. 7.4).

    Apenas a normalização é mantida, usando os MESMOS valores de média e
    desvio padrão do treino. Usar estatísticas do teste seria "data leakage"
    — uma violação fundamental que infla artificialmente as métricas
    (Kaufman et al., 2012, "Leakage in Data Mining").

    Parâmetros:
        normalize (bool): Se True, aplica normalização z-score.

    Retorna:
        torchvision.transforms.Compose ou None: Pipeline de transformações.
    """
    if normalize:
        return T.Compose([
            T.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ])
    return None


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverte a normalização z-score para visualização de imagens.

    Para exibir uma imagem que foi normalizada, precisamos desfazer a
    transformação: x_original = x_norm × std + mean.

    Isso é necessário porque matplotlib espera pixels no intervalo [0, 1]
    para imagens float, e imagens normalizadas podem ter valores negativos.

    Parâmetros:
        tensor (torch.Tensor): Imagem normalizada, shape (1, 28, 28) ou
                               (28, 28).

    Retorna:
        torch.Tensor: Imagem desnormalizada no intervalo [0, 1].
    """
    # Cria tensores de média e desvio padrão com broadcast compatível.
    mean = torch.tensor(MNIST_MEAN).view(-1, 1, 1)
    std  = torch.tensor(MNIST_STD).view(-1, 1, 1)

    # Aplica a inversão: x = x_norm * std + mean
    denorm = tensor * std + mean

    # Clamp garante que os valores fiquem em [0, 1], evitando artefatos
    # de visualização causados por imprecisão numérica de ponto flutuante.
    return torch.clamp(denorm, 0.0, 1.0)
