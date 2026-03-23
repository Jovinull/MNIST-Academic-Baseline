# ============================================================================
# MNIST Academic Baseline — Módulo de Treinamento
# ============================================================================
# Este módulo implementa o loop de treinamento e validação, o coração de
# qualquer pipeline de Deep Learning. O loop segue o paradigma de
# Aprendizado Supervisionado, onde o modelo ajusta seus parâmetros
# iterativamente para minimizar uma função de perda sobre dados rotulados.
#
# O fluxo de cada ÉPOCA é:
#   1. TREINO: para cada mini-batch, computa forward pass → perda →
#      backward pass (gradientes) → atualização dos pesos.
#   2. VALIDAÇÃO: avalia o modelo nos dados de validação SEM atualizar
#      pesos (mode=eval, no_grad). Monitora overfitting.
#
# Referências:
#   - Goodfellow, Bengio & Courville (2016), "Deep Learning", cap. 8 —
#     "Optimization for Training Deep Models".
#   - Bottou (2012), "Stochastic Gradient Descent Tricks", Neural
#     Networks: Tricks of the Trade.
#   - Smith (2018), "A Disciplined Approach to Neural Network
#     Hyper-Parameters", arXiv:1803.09820.
# ============================================================================

import time                        # Medição de tempo por época
import copy                        # Deep copy do melhor modelo
import logging                     # Logging estruturado

import torch                       # Framework de tensores
import torch.nn as nn              # Módulos de redes neurais
import torch.optim as optim        # Algoritmos de otimização
from torch.utils.data import DataLoader

from tqdm import tqdm              # Barras de progresso

# Configura o logger deste módulo.
logger = logging.getLogger(__name__)


# ============================================================================
# FUNÇÕES AUXILIARES DE CONSTRUÇÃO
# ============================================================================

def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    """
    Factory function que instancia o otimizador a partir do seu nome.

    OTIMIZADORES IMPLEMENTADOS:

    1. SGD com Momentum (Polyak, 1964; Sutskever et al., 2013):
       O Stochastic Gradient Descent atualiza os pesos na direção oposta
       ao gradiente da perda. O momentum acumula gradientes passados como
       uma "velocidade", acelerando a convergência em direções consistentes
       e amortecendo oscilações em direções instáveis.
       Equação: v_t = μ·v_{t-1} + g_t; θ_t = θ_{t-1} - lr·v_t
       onde μ=0.9 é o coeficiente de momentum.

    2. Adam (Kingma & Ba, 2015 — "Adam: A Method for Stochastic
       Optimization", ICLR):
       Combina os benefícios do Momentum (primeiro momento — média dos
       gradientes) com RMSProp (segundo momento — variância dos gradientes).
       Adapta a taxa de aprendizado individualmente para cada parâmetro,
       sendo robusto a escalonamento do gradiente e ruído.
       Equação: m_t = β₁·m_{t-1} + (1-β₁)·g_t  (média móvel)
                v_t = β₂·v_{t-1} + (1-β₂)·g_t² (variância móvel)
                θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)

    3. AdamW (Loshchilov & Hutter, 2019 — "Decoupled Weight Decay
       Regularization", ICLR):
       Corrige um problema sutil do Adam original: a regularização L2
       interage mal com a adaptação de taxa por parâmetro. O AdamW
       "desacopla" o weight decay da atualização adaptativa, aplicando
       a penalização diretamente aos pesos em vez de incluí-la no
       gradiente. Isso resulta em melhor generalização.

    Parâmetros:
        model (nn.Module): Modelo cujos parâmetros serão otimizados.
        optimizer_name (str): 'SGD', 'Adam' ou 'AdamW'.
        learning_rate (float): Taxa de aprendizado base.
        weight_decay (float): Coeficiente de regularização L2.

    Retorna:
        torch.optim.Optimizer: Otimizador configurado.
    """
    # Filtra apenas parâmetros treináveis (requires_grad=True).
    # Parâmetros congelados (e.g., em transfer learning) não devem
    # receber atualizações do otimizador.
    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == 'SGD':
        return optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,             # Valor padrão recomendado (Sutskever, 2013)
            weight_decay=weight_decay,
            nesterov=True,             # Nesterov momentum: "look-ahead" gradient
        )                              # melhora convergência (Nesterov, 1983)
    elif optimizer_name == 'Adam':
        return optim.Adam(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),        # Valores padrão do paper original
            eps=1e-8,                  # Estabilidade numérica na divisão
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'AdamW':
        return optim.AdamW(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay, # Weight decay DESACOPLADO
        )
    else:
        raise ValueError(
            f"Otimizador '{optimizer_name}' não suportado. "
            f"Opções: ['SGD', 'Adam', 'AdamW']"
        )


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    epochs: int,
):
    """
    Factory function que instancia o scheduler de learning rate.

    SCHEDULERS IMPLEMENTADOS:

    1. StepLR: decai a taxa por um fator gamma a cada 'step_size' épocas.
       Equação: lr_t = lr_0 × gamma^(floor(epoch / step_size))
       Simples e previsível. step_size = epochs//3 divide o treino em
       3 fases de aprendizado decrescente.

    2. CosineAnnealingLR (Loshchilov & Hutter, 2017 — "SGDR: Stochastic
       Gradient Descent with Warm Restarts"):
       A taxa segue uma curva cosseno de lr_max até lr_min (≈0).
       Equação: lr_t = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(πt/T))
       Vantagens:
         - Decaimento suave, sem "degraus" abruptos.
         - Períodos finais com lr muito baixo permitem ajuste fino dos
           pesos, convergindo para mínimos locais mais agudos (sharper).

    3. ReduceLROnPlateau: monitora uma métrica (val_loss) e reduz a lr
       quando a métrica para de melhorar por 'patience' épocas.
       Abordagem reativa: só ajusta quando necessário.
       Parâmetros:
         - patience=5: tolera 5 épocas sem melhoria antes de reduzir.
         - factor=0.5: reduz a lr pela metade a cada trigger.

    Parâmetros:
        optimizer (Optimizer): Otimizador ao qual o scheduler será anexado.
        scheduler_name (str): Nome do scheduler ou 'None' para desabilitar.
        epochs (int): Número total de épocas de treinamento.

    Retorna:
        Scheduler ou None: Instância do scheduler, ou None se desabilitado.
    """
    if scheduler_name == 'None' or scheduler_name is None:
        return None
    elif scheduler_name == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 3),  # Decai 3 vezes durante o treino
            gamma=0.1,                       # Reduz lr por fator de 10
        )
    elif scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,        # Período completo do cosseno = total de épocas
            eta_min=1e-6,        # lr mínima (evita lr exatamente zero)
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # Monitora minimização (da val_loss)
            factor=0.5,          # Reduz lr pela metade
            patience=5,          # Espera 5 épocas sem melhoria
            min_lr=1e-6,         # lr mínima permitida
            verbose=False,
        )
    else:
        raise ValueError(
            f"Scheduler '{scheduler_name}' não suportado. "
            f"Opções: ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'None']"
        )


# ============================================================================
# LOOP DE TREINAMENTO E VALIDAÇÃO
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Executa UMA ÉPOCA de treinamento sobre todo o conjunto de treino.

    Uma época = uma passagem completa por todos os mini-batches do dataset.
    Para cada mini-batch, o ciclo fundamental do aprendizado supervisionado
    é executado:

        1. FORWARD PASS:  ŷ = model(x)           — predições
        2. LOSS:          L = criterion(ŷ, y)     — erro quantificado
        3. BACKWARD PASS: L.backward()            — computa gradientes ∂L/∂θ
        4. UPDATE:        optimizer.step()         — atualiza θ ← θ - lr·∇L
        5. ZERO GRAD:     optimizer.zero_grad()    — limpa gradientes acumulados

    O passo 5 é crucial no PyTorch: por padrão, gradientes são ACUMULADOS
    (somados) a cada chamada de .backward(). Se não forem zerados, o
    gradiente usado na atualização seria a soma de todos os batches
    anteriores, o que é incorreto para SGD padrão.

    Parâmetros:
        model (nn.Module): Modelo a ser treinado.
        dataloader (DataLoader): DataLoader do conjunto de treino.
        criterion (nn.Module): Função de perda (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Algoritmo de otimização.
        device (torch.device): Dispositivo de computação (CPU ou GPU).
        epoch (int): Número da época atual (para logging).

    Retorna:
        dict: {'loss': float, 'accuracy': float} — métricas da época.
    """
    # Coloca o modelo em modo de TREINAMENTO.
    # Isso ativa comportamentos específicos de treino:
    #   - Dropout: neurônios são desativados aleatoriamente.
    #   - BatchNorm: usa estatísticas do mini-batch atual e atualiza
    #     as running statistics (média e variância acumuladas).
    model.train()

    # Acumuladores para métricas da época.
    running_loss    = 0.0   # Soma da perda de todos os batches
    correct         = 0     # Contagem de predições corretas
    total           = 0     # Contagem total de amostras processadas

    # Itera sobre todos os mini-batches do DataLoader com barra de progresso.
    progress_bar = tqdm(
        dataloader,
        desc=f"  Treino Época {epoch:>3d}",
        leave=False,
        ncols=100,
    )

    for batch_idx, (images, labels) in enumerate(progress_bar):
        # -----------------------------------------------------------------
        # TRANSFERÊNCIA PARA O DISPOSITIVO (CPU/GPU)
        # -----------------------------------------------------------------
        # .to(device) move os tensores para a memória do dispositivo alvo.
        # Se device='cuda', os dados são copiados para a memória da GPU
        # (VRAM), onde as operações matriciais são 10-100× mais rápidas
        # graças ao paralelismo massivo das GPUs.
        #
        # non_blocking=True permite transferência assíncrona quando
        # pin_memory=True no DataLoader, sobrepondo I/O com computação.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # -----------------------------------------------------------------
        # PASSO 1: ZERAR GRADIENTES
        # -----------------------------------------------------------------
        # Limpa os gradientes acumulados da iteração anterior.
        # set_to_none=True é mais eficiente que .zero_grad() padrão,
        # pois evita a operação de memset (preenchimento com zeros),
        # substituindo por uma desalocação dos tensores de gradiente.
        optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------
        # PASSO 2: FORWARD PASS
        # -----------------------------------------------------------------
        # Propaga os dados através da rede, computando as ativações de
        # cada camada e produzindo os logits de saída.
        # O PyTorch automaticamente constrói o GRAFO COMPUTACIONAL
        # (computational graph) durante o forward pass, registrando
        # cada operação para que os gradientes possam ser calculados
        # automaticamente via regra da cadeia (chain rule).
        outputs = model(images)

        # -----------------------------------------------------------------
        # PASSO 3: CÁLCULO DA PERDA
        # -----------------------------------------------------------------
        # CrossEntropyLoss combina LogSoftmax + NLLLoss em uma única
        # operação numericamente estável.
        #
        # Internamente:
        #   1. LogSoftmax: log(exp(x_i) / Σ_j exp(x_j))
        #      Converte logits em log-probabilidades.
        #   2. NLLLoss: -log(p_{y_true})
        #      Negative Log-Likelihood: penaliza a probabilidade atribuída
        #      à classe correta.
        #
        # A perda é a MÉDIA sobre as amostras do batch (reduction='mean').
        loss = criterion(outputs, labels)

        # -----------------------------------------------------------------
        # PASSO 4: BACKWARD PASS (BACKPROPAGATION)
        # -----------------------------------------------------------------
        # Calcula os gradientes de TODOS os parâmetros treináveis em
        # relação à perda, percorrendo o grafo computacional de trás
        # para frente (da perda até os inputs).
        #
        # Algoritmo: regra da cadeia generalizada (chain rule).
        # Para cada parâmetro θ: ∂L/∂θ = ∂L/∂a_n × ∂a_n/∂a_{n-1} × ... × ∂a_1/∂θ
        # onde a_i são as ativações intermediárias.
        #
        # Os gradientes são armazenados em param.grad para cada parâmetro.
        loss.backward()

        # -----------------------------------------------------------------
        # PASSO 5: ATUALIZAÇÃO DOS PESOS
        # -----------------------------------------------------------------
        # O otimizador usa os gradientes calculados para atualizar os
        # pesos do modelo na direção que minimiza a perda.
        # A regra exata depende do otimizador (SGD, Adam, etc.).
        optimizer.step()

        # -----------------------------------------------------------------
        # ACUMULAÇÃO DE MÉTRICAS
        # -----------------------------------------------------------------
        # loss.item() extrai o valor escalar do tensor, liberando o
        # grafo computacional da memória.
        # images.size(0) é o tamanho do batch atual (pode ser menor
        # que batch_size no último batch da época).
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # torch.argmax retorna o índice da classe com maior logit.
        # Comparar com o rótulo verdadeiro dá o número de acertos.
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total   += batch_size

        # Atualiza a barra de progresso com métricas em tempo real.
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc':  f"{100.0 * correct / total:.2f}%",
        })

    # Calcula métricas médias da época inteira.
    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Avalia o modelo no conjunto de validação ou teste.

    DIFERENÇAS FUNDAMENTAIS EM RELAÇÃO AO TREINAMENTO:

    1. model.eval(): ativa o modo de avaliação, que:
       - Dropout: DESATIVADO (todos os neurônios são usados).
       - BatchNorm: usa as running statistics (média e variância
         acumuladas durante o treino) em vez de estatísticas do batch.

    2. torch.no_grad(): desabilita o cálculo de gradientes.
       - Reduz o consumo de memória (não precisa armazenar ativações
         intermediárias para backpropagation).
       - Acelera a computação (pula operações de autograd).
       - NUNCA deve ser omitido na validação/teste, pois o vazamento
         de gradientes poderia corromper os parâmetros do modelo.

    Parâmetros:
        model (nn.Module): Modelo a ser avaliado.
        dataloader (DataLoader): DataLoader de validação ou teste.
        criterion (nn.Module): Função de perda.
        device (torch.device): Dispositivo de computação.

    Retorna:
        dict: {'loss': float, 'accuracy': float} — métricas de avaliação.
    """
    # Ativa modo de avaliação (desliga dropout e fixa BatchNorm).
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    # torch.no_grad() é um context manager que desabilita o autograd.
    # Todas as operações dentro deste bloco NÃO constroem grafo
    # computacional, economizando memória e tempo.
    with torch.no_grad():
        for images, labels in dataloader:
            # Move dados para o dispositivo alvo.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass APENAS (sem backward).
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Acumulação de métricas.
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total   += batch_size

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


# ============================================================================
# TREINADOR COMPLETO (Orquestrador do Pipeline)
# ============================================================================

class Trainer:
    """
    Classe que orquestra o pipeline completo de treinamento e validação.

    Responsabilidades:
    1. Gerenciar o loop de épocas (treino + validação).
    2. Aplicar o scheduler de learning rate.
    3. Implementar Early Stopping para prevenir overfitting.
    4. Salvar o melhor modelo (checkpoint) baseado na métrica de validação.
    5. Registrar o histórico de métricas para análise posterior.

    EARLY STOPPING (Prechelt, 1998 — "Early Stopping — But When?"):
    Monitora a perda de validação e interrompe o treinamento se ela não
    melhorar por 'patience' épocas consecutivas. Isso previne que o modelo
    continue "memorizando" o conjunto de treino após ter atingido sua
    melhor generalização. O modelo salvo é o do melhor epoch, não do último.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler=None,
        patience: int = 7,
        checkpoint_path: str = None,
    ):
        """
        Inicializa o Trainer com todos os componentes do pipeline.

        Parâmetros:
            model (nn.Module): Modelo a ser treinado.
            optimizer (Optimizer): Algoritmo de otimização.
            criterion (nn.Module): Função de perda.
            device (torch.device): Dispositivo alvo (CPU/GPU).
            scheduler: Learning rate scheduler (opcional).
            patience (int): Épocas de tolerância para early stopping.
            checkpoint_path (str): Caminho para salvar o melhor modelo.
        """
        self.model           = model.to(device)   # Move modelo para GPU/CPU
        self.optimizer        = optimizer
        self.criterion        = criterion
        self.device           = device
        self.scheduler        = scheduler
        self.patience         = patience
        self.checkpoint_path  = checkpoint_path

        # Estado do early stopping.
        self.best_val_loss    = float('inf')  # Inicializa com +∞
        self.best_val_acc     = 0.0
        self.epochs_no_improve = 0            # Contador de épocas sem melhoria
        self.best_model_state = None          # Pesos do melhor modelo

        # Histórico de métricas para cada época (para gráficos posteriores).
        self.history = {
            'train_loss': [],
            'train_acc':  [],
            'val_loss':   [],
            'val_acc':    [],
            'lr':         [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        trial=None,
    ) -> dict:
        """
        Executa o loop completo de treinamento por N épocas.

        Este método implementa o ciclo externo do treinamento:
        Para cada época → treina → valida → scheduler → early stopping.

        O parâmetro 'trial' (Optuna) permite integração com o módulo de
        tuning: a cada época, reporta a métrica intermediária ao Optuna,
        que pode decidir podar (prune) trials pouco promissores.

        Parâmetros:
            train_loader (DataLoader): Dados de treinamento.
            val_loader (DataLoader): Dados de validação.
            epochs (int): Número máximo de épocas.
            trial: Objeto Optuna Trial para pruning (opcional).

        Retorna:
            dict: Histórico completo de métricas de todas as épocas.
        """
        logger.info(f"Iniciando treinamento por {epochs} épocas no {self.device}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # ----- FASE DE TREINAMENTO -----
            train_metrics = train_one_epoch(
                self.model, train_loader, self.criterion,
                self.optimizer, self.device, epoch,
            )

            # ----- FASE DE VALIDAÇÃO -----
            val_metrics = validate(
                self.model, val_loader, self.criterion, self.device,
            )

            # Captura a learning rate atual (pode mudar via scheduler).
            current_lr = self.optimizer.param_groups[0]['lr']

            # ----- ATUALIZAÇÃO DO SCHEDULER -----
            if self.scheduler is not None:
                # ReduceLROnPlateau requer a métrica monitorada.
                if isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Registra métricas no histórico.
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Log formatado para monitoramento durante o treinamento.
            logger.info(
                f"Época {epoch:>3d}/{epochs} | "
                f"Treino Loss: {train_metrics['loss']:.4f} | "
                f"Treino Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tempo: {epoch_time:.1f}s"
            )

            # ----- EARLY STOPPING + CHECKPOINT -----
            # Verifica se houve melhoria na perda de validação.
            if val_metrics['loss'] < self.best_val_loss:
                # Melhoria encontrada: reseta o contador e salva o modelo.
                self.best_val_loss     = val_metrics['loss']
                self.best_val_acc      = val_metrics['accuracy']
                self.epochs_no_improve = 0

                # Deep copy dos pesos do modelo (state_dict).
                # Precisamos de uma cópia PROFUNDA porque os pesos são
                # modificados in-place durante o treinamento.
                self.best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

                # Salva checkpoint em disco (se caminho especificado).
                if self.checkpoint_path:
                    torch.save({
                        'epoch':      epoch,
                        'model_state': self.best_model_state,
                        'optimizer_state': self.optimizer.state_dict(),
                        'val_loss':   self.best_val_loss,
                        'val_acc':    self.best_val_acc,
                    }, self.checkpoint_path)
                    logger.info(f"  ✓ Melhor modelo salvo (val_acc: {self.best_val_acc:.4f})")
            else:
                # Sem melhoria: incrementa o contador de paciência.
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logger.info(
                        f"  Early stopping ativado após {self.patience} "
                        f"épocas sem melhoria."
                    )
                    break

            # ----- INTEGRAÇÃO COM OPTUNA (pruning) -----
            # Se um trial do Optuna foi passado, reporta a métrica
            # intermediária. O Optuna pode decidir podar este trial
            # se ele estiver significativamente abaixo da mediana dos
            # outros trials na mesma época.
            if trial is not None:
                import optuna
                trial.report(val_metrics['accuracy'], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Ao final do treinamento, restaura os pesos do melhor modelo.
        # Isso garante que o modelo retornado é o da época com melhor
        # generalização, não o da última época (que pode ter overfitting).
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"Modelo restaurado para melhor época "
                f"(val_loss: {self.best_val_loss:.4f}, "
                f"val_acc: {self.best_val_acc:.4f})"
            )

        return self.history
