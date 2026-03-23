# ============================================================================
# MNIST Academic Baseline — Módulo de Arquiteturas de Redes Neurais
# ============================================================================
# Este módulo define quatro arquiteturas progressivamente mais sofisticadas
# para classificação de dígitos manuscritos, cada uma representando um
# marco importante na evolução do campo de Deep Learning:
#
#   1. MLP (Multilayer Perceptron):
#      Baseline com camadas totalmente conectadas. Demonstra que redes
#      feedforward simples já são capazes de aprender representações
#      úteis, mas ignoram a estrutura espacial 2D das imagens.
#      Referência: Rumelhart, Hinton & Williams (1986) — "Learning
#      representations by back-propagating errors", Nature.
#
#   2. LeNet-5:
#      Arquitetura convolucional pioneira, projetada especificamente para
#      reconhecimento de dígitos manuscritos. Introduziu o conceito de
#      campos receptivos locais e compartilhamento de pesos.
#      Referência: LeCun, Bottou, Bengio & Haffner (1998) — "Gradient-Based
#      Learning Applied to Document Recognition", Proc. IEEE.
#
#   3. ModernCNN:
#      Evolução da LeNet com técnicas modernas de regularização:
#      Batch Normalization (Ioffe & Szegedy, 2015) e Dropout
#      (Srivastava et al., 2014). Usa ReLU em vez de tanh/sigmoid.
#
#   4. DeepCNN:
#      Arquitetura profunda inspirada em blocos residuais (He et al., 2016).
#      Demonstra como conexões residuais permitem treinar redes mais
#      profundas ao mitigar o problema de vanishing gradients.
#
# NOTA SOBRE A PROGRESSÃO PEDAGÓGICA:
# A ordem das arquiteturas reflete a evolução histórica das CNNs e serve
# como um tour guiado pelas inovações-chave do campo. Ao comparar os
# resultados de cada arquitetura no mesmo dataset, o estudante pode
# quantificar o impacto de cada inovação arquitetural.
# ============================================================================

import torch
import torch.nn as nn              # Módulo de construção de redes neurais
import torch.nn.functional as F    # Funções funcionais (ativações, pooling)


# ============================================================================
# 1. MULTILAYER PERCEPTRON (MLP) — Baseline Fully-Connected
# ============================================================================

class MLP(nn.Module):
    """
    Rede neural feedforward com camadas totalmente conectadas (Dense).

    Arquitetura:
        Input (784) → FC(512) → ReLU → Dropout → FC(256) → ReLU → Dropout
        → FC(10) → Softmax (implícito no CrossEntropyLoss)

    O MLP serve como baseline por três razões:
    1. É a arquitetura mais simples que pode aprender o MNIST (~98% acc).
    2. Ao achatar a imagem 28×28 em um vetor de 784 dimensões, ele perde
       toda a informação espacial (adjacência de pixels), o que o torna
       inferior às CNNs para tarefas visuais.
    3. Tem muito mais parâmetros que uma CNN equivalente (sem compartilhamento
       de pesos), o que o torna propenso a overfitting.

    Referências:
        - Goodfellow et al. (2016), cap. 6 — "Deep Feedforward Networks".
        - Hornik (1991), "Approximation capabilities of multilayer feedforward
          networks" — teorema da aproximação universal.
    """

    def __init__(self, dropout_rate: float = 0.25):
        """
        Inicializa as camadas do MLP.

        Parâmetros:
            dropout_rate (float): Probabilidade de desativar neurônios.
                                  Valor padrão 0.25 (recomendação de
                                  Srivastava et al., 2014, para camadas
                                  ocultas de redes feedforward).
        """
        # Chama o construtor da classe pai nn.Module, que inicializa o
        # registro interno de parâmetros treináveis do PyTorch.
        super(MLP, self).__init__()

        # -----------------------------------------------------------------
        # FLATTEN: Converte imagens (1, 28, 28) → vetores (784,)
        # -----------------------------------------------------------------
        # nn.Flatten() achata todas as dimensões exceto a do batch.
        # Uma imagem 28×28 de 1 canal = 28 × 28 × 1 = 784 features.
        self.flatten = nn.Flatten()

        # -----------------------------------------------------------------
        # CAMADA OCULTA 1: 784 → 512 neurônios
        # -----------------------------------------------------------------
        # nn.Linear implementa a transformação afim: y = xW^T + b
        # onde W ∈ ℝ^{512×784} e b ∈ ℝ^{512}.
        #
        # Escolha de 512 neurônios: potência de 2 (eficiência em GPU),
        # grande o suficiente para capturar padrões complexos, mas pequena
        # o bastante para evitar overfitting excessivo no MNIST.
        self.fc1 = nn.Linear(28 * 28, 512)

        # -----------------------------------------------------------------
        # CAMADA OCULTA 2: 512 → 256 neurônios
        # -----------------------------------------------------------------
        # Redução gradual da dimensionalidade (funil) é uma prática padrão
        # que força a rede a aprender representações cada vez mais
        # compactas e abstratas (Bengio et al., 2013, "Representation
        # Learning").
        self.fc2 = nn.Linear(512, 256)

        # -----------------------------------------------------------------
        # CAMADA DE SAÍDA: 256 → 10 classes
        # -----------------------------------------------------------------
        # O MNIST tem 10 classes (dígitos 0-9). A saída são 10 logits
        # (valores reais não normalizados). A normalização em probabilidades
        # é feita internamente pelo CrossEntropyLoss (que aplica LogSoftmax).
        self.fc3 = nn.Linear(256, 10)

        # -----------------------------------------------------------------
        # DROPOUT: Regularização estocástica
        # -----------------------------------------------------------------
        # Durante o treinamento, cada neurônio é desativado (zerado) com
        # probabilidade p a cada forward pass. Isso:
        #   1. Previne co-adaptação entre neurônios.
        #   2. Funciona como um ensemble implícito de sub-redes.
        #   3. Equivale aproximadamente a regularização L2 (Wager et al., 2013).
        #
        # Na inferência, dropout é desativado automaticamente (model.eval())
        # e as ativações são escaladas por (1-p) para compensar.
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagação direta (forward pass) do MLP.

        Define o fluxo computacional: como os dados fluem da entrada até
        a saída, passando por cada camada sequencialmente.

        Parâmetros:
            x (torch.Tensor): Batch de imagens, shape (B, 1, 28, 28).

        Retorna:
            torch.Tensor: Logits de classe, shape (B, 10).
        """
        # Achata a imagem 2D em um vetor 1D: (B, 1, 28, 28) → (B, 784)
        x = self.flatten(x)

        # Camada 1: transformação linear + ativação não-linear + dropout.
        # ReLU(x) = max(0, x) — a ativação mais usada em redes profundas.
        # Vantagens sobre sigmoid/tanh:
        #   - Gradiente não satura para x > 0 (mitiga vanishing gradient).
        #   - Computacionalmente eficiente (apenas comparação com zero).
        # Referência: Glorot et al. (2011), "Deep Sparse Rectifier Neural Networks".
        x = self.dropout(F.relu(self.fc1(x)))

        # Camada 2: mesma sequência Linear → ReLU → Dropout.
        x = self.dropout(F.relu(self.fc2(x)))

        # Camada de saída: apenas transformação linear (sem ativação).
        # Os logits brutos serão processados pelo CrossEntropyLoss,
        # que internamente aplica LogSoftmax para estabilidade numérica.
        x = self.fc3(x)

        return x


# ============================================================================
# 2. LeNet-5 — CNN Clássica (LeCun et al., 1998)
# ============================================================================

class LeNet5(nn.Module):
    """
    Implementação modernizada da LeNet-5, adaptada para imagens 28×28.

    A LeNet-5 original usava ativações tanh e subsampling por média.
    Esta versão usa ReLU e max pooling, que são o padrão moderno.

    Arquitetura:
        Conv(1→6, 5×5) → ReLU → MaxPool(2×2) →
        Conv(6→16, 5×5) → ReLU → MaxPool(2×2) →
        Flatten → FC(256→120) → ReLU → Dropout →
        FC(120→84) → ReLU → Dropout → FC(84→10)

    Dimensões dos feature maps:
        (B, 1, 28, 28) → Conv1 → (B, 6, 24, 24) → Pool → (B, 6, 12, 12)
        → Conv2 → (B, 16, 8, 8) → Pool → (B, 16, 4, 4) → Flatten → (B, 256)

    Conceitos-chave demonstrados:
    1. CAMPOS RECEPTIVOS LOCAIS: cada filtro convolucional opera sobre uma
       região espacial limitada (5×5), capturando padrões locais (bordas,
       cantos, curvas) sem precisar ver a imagem inteira.
    2. COMPARTILHAMENTO DE PESOS: o mesmo filtro é aplicado em todas as
       posições da imagem, reduzindo drasticamente o número de parâmetros
       em relação a um MLP equivalente.
    3. HIERARQUIA DE FEATURES: camadas iniciais detectam features simples
       (bordas), camadas profundas combinam features em padrões complexos
       (traços, arcos, loops de dígitos).

    Referência:
        LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
        "Gradient-Based Learning Applied to Document Recognition."
        Proceedings of the IEEE, 86(11), 2278-2324.
    """

    def __init__(self, dropout_rate: float = 0.25):
        """
        Inicializa as camadas convolucionais e fully-connected da LeNet-5.

        Parâmetros:
            dropout_rate (float): Taxa de dropout para as camadas FC.
        """
        super(LeNet5, self).__init__()

        # -----------------------------------------------------------------
        # BLOCO CONVOLUCIONAL 1
        # -----------------------------------------------------------------
        # Conv2d(in_channels=1, out_channels=6, kernel_size=5):
        #   - in_channels=1: imagem grayscale (1 canal de cor).
        #   - out_channels=6: produz 6 feature maps (6 filtros aprendem
        #     6 padrões diferentes, e.g., bordas horizontais, verticais,
        #     diagonais, cantos).
        #   - kernel_size=5: cada filtro tem tamanho 5×5 pixels, criando
        #     um campo receptivo local de 25 pixels.
        #
        # Sem padding: a saída encolhe de 28×28 para 24×24.
        # Fórmula: output_size = (input_size - kernel_size) / stride + 1
        #                       = (28 - 5) / 1 + 1 = 24
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )

        # -----------------------------------------------------------------
        # BLOCO CONVOLUCIONAL 2
        # -----------------------------------------------------------------
        # Conv2d(6→16, 5×5): 16 filtros operam sobre os 6 feature maps da
        # camada anterior, combinando features de baixo nível em
        # representações mais abstratas.
        #
        # Após pooling do bloco 1: 12×12 → Conv(5×5) → 8×8
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )

        # -----------------------------------------------------------------
        # CAMADAS FULLY-CONNECTED
        # -----------------------------------------------------------------
        # Após conv2 + pool: shape = (B, 16, 4, 4) → flatten → (B, 256)
        # A transição conv→FC conecta as features espaciais extraídas
        # pelas convoluções ao classificador final.
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 120 neurônios (original)
        self.fc2 = nn.Linear(120, 84)            # 84 neurônios (original)
        self.fc3 = nn.Linear(84, 10)             # 10 classes de saída

        # Dropout aplicado apenas nas camadas FC (não nas convolucionais).
        # Convoluções já têm regularização implícita via compartilhamento
        # de pesos e campos receptivos locais.
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da LeNet-5.

        Parâmetros:
            x (torch.Tensor): Batch de imagens, shape (B, 1, 28, 28).

        Retorna:
            torch.Tensor: Logits de classe, shape (B, 10).
        """
        # BLOCO 1: Convolução → Ativação → Pooling
        # Conv: (B, 1, 28, 28) → (B, 6, 24, 24)
        # ReLU: ativação element-wise, mantém shape
        # MaxPool(2, 2): reduz resolução pela metade → (B, 6, 12, 12)
        #   Max pooling seleciona o valor máximo em cada janela 2×2.
        #   Isso provê invariância a pequenas translações e reduz
        #   a dimensionalidade computacional (Scherer et al., 2010).
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)

        # BLOCO 2: mesma sequência Conv → ReLU → MaxPool
        # Conv: (B, 6, 12, 12) → (B, 16, 8, 8)
        # Pool: (B, 16, 8, 8) → (B, 16, 4, 4)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)

        # FLATTEN: (B, 16, 4, 4) → (B, 256)
        # Converte o tensor 3D de features espaciais em vetor 1D para
        # alimentar as camadas fully-connected.
        x = x.view(x.size(0), -1)

        # CLASSIFICADOR: FC → ReLU → Dropout → FC → ReLU → Dropout → FC
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # Logits (sem ativação final)

        return x


# ============================================================================
# 3. ModernCNN — CNN com Batch Normalization e Dropout
# ============================================================================

class ModernCNN(nn.Module):
    """
    CNN moderna que incorpora técnicas de regularização pós-2015.

    Inovações em relação à LeNet-5:

    1. BATCH NORMALIZATION (Ioffe & Szegedy, 2015):
       Normaliza as ativações de cada camada usando estatísticas do
       mini-batch, reduzindo o "Internal Covariate Shift". Benefícios:
         - Permite taxas de aprendizado maiores.
         - Atua como regularizador (reduz necessidade de dropout).
         - Acelera a convergência em 5-14× (resultado original).

    2. MAIS FILTROS CONVOLUCIONAIS:
       A largura (número de canais) cresce progressivamente: 32 → 64,
       seguindo a heurística de duplicar canais a cada pooling
       (Simonyan & Zisserman, 2015 — VGGNet).

    3. DROPOUT EM CAMADAS CONVOLUCIONAIS:
       Dropout2d (Spatial Dropout) desativa feature maps inteiros em vez
       de neurônios individuais, o que é mais eficaz para dados espaciais
       (Tompson et al., 2015, "Efficient Object Localization Using
       Convolutional Networks").

    Arquitetura:
        Conv(1→32, 3×3) → BN → ReLU → Conv(32→32, 3×3) → BN → ReLU →
        MaxPool(2×2) → Dropout2d →
        Conv(32→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU →
        MaxPool(2×2) → Dropout2d →
        Flatten → FC(1600→256) → BN1d → ReLU → Dropout → FC(256→10)
    """

    def __init__(self, dropout_rate: float = 0.25):
        """
        Inicializa a ModernCNN com BatchNorm e Dropout em todas as camadas.

        Parâmetros:
            dropout_rate (float): Taxa de dropout. Aplicada de forma
                                  diferenciada: metade nas convoluções
                                  (Dropout2d), valor completo nas FC.
        """
        super(ModernCNN, self).__init__()

        # -----------------------------------------------------------------
        # BLOCO CONVOLUCIONAL 1 (2 convoluções empilhadas)
        # -----------------------------------------------------------------
        # Empilhar convoluções 3×3 sem pooling intermediário aumenta o
        # campo receptivo efetivo sem perder resolução prematuramente.
        # Duas convoluções 3×3 têm campo receptivo equivalente a uma 5×5,
        # mas com menos parâmetros e mais não-linearidades (Simonyan &
        # Zisserman, 2015 — VGGNet).
        #
        # nn.Sequential agrupa as operações em um sub-módulo coeso.
        self.block1 = nn.Sequential(
            # Conv 1a: (B, 1, 28, 28) → (B, 32, 26, 26)
            # kernel_size=3, padding=0: output = (28-3)/1 + 1 = 26
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # BatchNorm2d normaliza cada canal independentemente.
            # Para 32 canais, aprende 32 pares (γ, β) de parâmetros
            # de escala e deslocamento.
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # inplace=True economiza memória

            # Conv 1b: (B, 32, 28, 28) → (B, 32, 28, 28) com padding=1
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # MaxPool: (B, 32, 28, 28) → (B, 32, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dropout2d: desativa feature maps inteiros com prob p/2.
            # Usar metade da taxa do dropout regular é uma heurística
            # comum, pois desativar um feature map inteiro tem impacto
            # maior do que desativar um único neurônio.
            nn.Dropout2d(p=dropout_rate / 2),
        )

        # -----------------------------------------------------------------
        # BLOCO CONVOLUCIONAL 2 (canais duplicados: 32 → 64)
        # -----------------------------------------------------------------
        # Duplicar canais a cada redução espacial compensa a perda de
        # informação espacial com maior capacidade representacional
        # (mais filtros = mais padrões detectáveis).
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # MaxPool: (B, 64, 14, 14) → (B, 64, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate / 2),
        )

        # -----------------------------------------------------------------
        # CLASSIFICADOR (Fully-Connected)
        # -----------------------------------------------------------------
        # Após block2: shape = (B, 64, 7, 7) → flatten → (B, 3136)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            # BatchNorm1d para camadas FC: normaliza o vetor de features.
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da ModernCNN.

        O fluxo é limpo e modular graças ao uso de nn.Sequential:
        cada bloco é uma unidade auto-contida.

        Parâmetros:
            x (torch.Tensor): Shape (B, 1, 28, 28).

        Retorna:
            torch.Tensor: Logits, shape (B, 10).
        """
        x = self.block1(x)       # Features de baixo nível
        x = self.block2(x)       # Features de alto nível
        x = self.classifier(x)   # Classificação
        return x


# ============================================================================
# 4. DeepCNN — CNN Profunda com Conexões Residuais
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Bloco residual (He et al., 2016 — "Deep Residual Learning for Image
    Recognition", CVPR, Best Paper Award).

    A ideia central é aprender a função RESIDUAL F(x) = H(x) - x, em vez
    da função desejada H(x) diretamente. A saída é então:

        H(x) = F(x) + x    (skip connection / atalho)

    Por que isso funciona?
    1. VANISHING GRADIENTS: em redes muito profundas, os gradientes
       diminuem exponencialmente durante a backpropagation, impedindo
       o aprendizado das camadas iniciais. A skip connection cria um
       caminho direto para o gradiente fluir, bypassing as camadas
       intermediárias:  ∂L/∂x = ∂L/∂H × (∂F/∂x + 1)
       O termo "+1" garante que o gradiente nunca se anula completamente.

    2. IDENTITY MAPPING: se as camadas internas aprenderem F(x) ≈ 0, o
       bloco se reduz à identidade H(x) = x. Isso permite que camadas
       "desnecessárias" simplesmente passem a informação adiante sem
       degradação, o que explica por que redes mais profundas nunca
       performam PIOR que redes rasas equivalentes (em teoria).

    Arquitetura do bloco:
        Input(x) → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    """

    def __init__(self, channels: int):
        """
        Inicializa um bloco residual com duas convoluções.

        Parâmetros:
            channels (int): Número de canais de entrada E saída.
                           A skip connection exige que as dimensões
                           de entrada e saída sejam idênticas.
        """
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # Primeira convolução: preserva dimensões com padding=1.
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # bias=False porque BatchNorm já aprende um bias (β),
            # tornar o bias da conv redundante desperdiçaria parâmetros
            # (Ioffe & Szegedy, 2015).
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # Segunda convolução: mesma configuração.
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            # NOTA: ReLU é aplicada APÓS a soma com o residual (ver forward).
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco residual.

        Implementa a equação fundamental: H(x) = F(x) + x

        Parâmetros:
            x (torch.Tensor): Feature map de entrada.

        Retorna:
            torch.Tensor: Feature map com skip connection aplicada.
        """
        # F(x): saída das duas convoluções
        residual = self.block(x)

        # H(x) = F(x) + x: adição element-wise (skip connection)
        # Esta operação exige que F(x) e x tenham exatamente o mesmo shape.
        out = residual + x

        # ReLU final aplicada APÓS a soma, seguindo a formulação original
        # de He et al. (2016). Aplicar ReLU antes da soma (pre-activation)
        # foi explorado posteriormente em He et al. (2016, v2), mas a
        # formulação pós-ativação é mais simples e igualmente eficaz para
        # redes de profundidade moderada.
        return F.relu(out, inplace=True)


class DeepCNN(nn.Module):
    """
    CNN profunda com blocos residuais, inspirada na ResNet.

    Esta arquitetura demonstra como as conexões residuais permitem
    aumentar a profundidade da rede (e consequentemente sua capacidade
    representacional) sem sofrer degradação de desempenho.

    Arquitetura:
        Stem: Conv(1→32, 3×3) → BN → ReLU →
        Stage1: ResBlock(32) → ResBlock(32) → MaxPool → Dropout2d →
        Transition: Conv(32→64, 3×3) → BN → ReLU →
        Stage2: ResBlock(64) → ResBlock(64) → MaxPool → Dropout2d →
        Classifier: AdaptiveAvgPool(1×1) → Flatten → FC(64→128) →
                     BN1d → ReLU → Dropout → FC(128→10)

    Total de camadas convolucionais: 1 (stem) + 4 (stage1) + 1 (trans) +
    4 (stage2) = 10 camadas convolucionais.

    Referência:
        He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual
        Learning for Image Recognition." CVPR. arXiv:1512.03385.
    """

    def __init__(self, dropout_rate: float = 0.25):
        """
        Inicializa a DeepCNN com blocos residuais e pooling adaptativo.

        Parâmetros:
            dropout_rate (float): Taxa de dropout.
        """
        super(DeepCNN, self).__init__()

        # -----------------------------------------------------------------
        # STEM (caule): primeira convolução que reduz o input bruto em
        # feature maps iniciais.
        # -----------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # -----------------------------------------------------------------
        # ESTÁGIO 1: dois blocos residuais com 32 canais.
        # -----------------------------------------------------------------
        # Cada ResidualBlock contém 2 convoluções → total de 4 camadas.
        # Não há redução de resolução dentro dos blocos residuais;
        # a redução é feita pelo MaxPool após o estágio.
        self.stage1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28×28 → 14×14
            nn.Dropout2d(p=dropout_rate / 2),
        )

        # -----------------------------------------------------------------
        # TRANSIÇÃO: aumenta o número de canais de 32 → 64.
        # -----------------------------------------------------------------
        # A transição é necessária porque os blocos residuais exigem
        # dimensões iguais na entrada e saída. Para mudar o número de
        # canais, usamos uma convolução 1×1 (pointwise convolution),
        # técnica popularizada pela Network in Network (Lin et al., 2014)
        # e pela Inception (Szegedy et al., 2015).
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -----------------------------------------------------------------
        # ESTÁGIO 2: dois blocos residuais com 64 canais.
        # -----------------------------------------------------------------
        self.stage2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14×14 → 7×7
            nn.Dropout2d(p=dropout_rate / 2),
        )

        # -----------------------------------------------------------------
        # CLASSIFICADOR com Adaptive Average Pooling
        # -----------------------------------------------------------------
        # AdaptiveAvgPool2d(1) reduz qualquer feature map para 1×1,
        # computando a média global de cada canal. Isso:
        #   1. Elimina a dependência do tamanho espacial do input.
        #   2. Atua como regularizador forte (Lin et al., 2014).
        #   3. Substitui a camada FC grande (Flatten(64*7*7)=3136 → FC)
        #      por um vetor compacto de 64 dimensões.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (B, 64, 7, 7) → (B, 64, 1, 1)
            nn.Flatten(),                  # (B, 64, 1, 1) → (B, 64)
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da DeepCNN.

        Parâmetros:
            x (torch.Tensor): Shape (B, 1, 28, 28).

        Retorna:
            torch.Tensor: Logits, shape (B, 10).
        """
        x = self.stem(x)          # Extração inicial de features
        x = self.stage1(x)        # Features com residuais (32 canais)
        x = self.transition(x)    # Expansão de canais: 32 → 64
        x = self.stage2(x)        # Features com residuais (64 canais)
        x = self.classifier(x)    # Classificação final
        return x


# ============================================================================
# FACTORY FUNCTION — Seleção de Arquitetura por Nome
# ============================================================================

# Dicionário que mapeia nomes (strings) às classes de arquitetura.
# Este padrão (Registry Pattern) permite selecionar arquiteturas
# dinamicamente via configuração YAML, sem modificar o código.
ARCHITECTURE_REGISTRY = {
    'MLP':       MLP,
    'LeNet5':    LeNet5,
    'ModernCNN': ModernCNN,
    'DeepCNN':  DeepCNN,
}


def build_model(architecture: str, dropout_rate: float = 0.25) -> nn.Module:
    """
    Factory function que instancia um modelo a partir do seu nome.

    Este padrão de projeto (Factory Method — Gamma et al., 1994,
    "Design Patterns") desacopla a seleção da arquitetura da sua
    instanciação, permitindo que o módulo de tuning itere sobre
    diferentes arquiteturas de forma programática.

    Parâmetros:
        architecture (str): Nome da arquitetura ('MLP', 'LeNet5',
                            'ModernCNN' ou 'DeepCNN').
        dropout_rate (float): Taxa de dropout passada ao construtor.

    Retorna:
        nn.Module: Modelo instanciado e pronto para treinamento.

    Raises:
        ValueError: Se o nome da arquitetura não estiver no registro.
    """
    if architecture not in ARCHITECTURE_REGISTRY:
        available = list(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(
            f"Arquitetura '{architecture}' não encontrada. "
            f"Opções disponíveis: {available}"
        )

    # Instancia a classe correspondente ao nome.
    model_class = ARCHITECTURE_REGISTRY[architecture]
    model = model_class(dropout_rate=dropout_rate)

    return model


def count_parameters(model: nn.Module) -> int:
    """
    Conta o número total de parâmetros treináveis do modelo.

    Parâmetros treináveis são aqueles com requires_grad=True.
    BatchNorm tem parâmetros treináveis (γ, β), mas buffers não
    treináveis (running mean, running var).

    Útil para comparar a complexidade relativa das arquiteturas
    e estimar os requisitos de memória do treinamento.

    Parâmetros:
        model (nn.Module): Modelo PyTorch.

    Retorna:
        int: Número total de parâmetros treináveis.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
