# ============================================================================
# MNIST Academic Baseline — Módulo de Otimização de Hiperparâmetros
# ============================================================================
# Este módulo implementa a busca automatizada de hiperparâmetros usando o
# framework Optuna, com suporte a três estratégias de otimização:
#
# 1. RANDOM SEARCH (Bergstra & Bengio, 2012):
#    Amostra hiperparâmetros aleatoriamente do espaço de busca.
#    Surpreendentemente eficaz: Bergstra & Bengio (2012) demonstraram que
#    Random Search supera Grid Search na maioria dos cenários, pois cada
#    trial explora uma combinação ÚNICA de valores, enquanto Grid Search
#    repete valores de parâmetros irrelevantes.
#
# 2. TPE — Tree-structured Parzen Estimator (Bergstra et al., 2011):
#    O sampler padrão do Optuna. Constrói um modelo probabilístico do
#    espaço de busca usando dois modelos de densidade:
#      - l(x): densidade dos hiperparâmetros nos "bons" trials (top γ%).
#      - g(x): densidade dos hiperparâmetros nos "maus" trials.
#    Seleciona novos pontos maximizando l(x)/g(x), concentrando a busca
#    nas regiões mais promissoras. Isso é uma forma de otimização
#    bayesiana sequencial baseada em modelo (SMBO).
#
# 3. GRID SEARCH (implementação via Optuna GridSampler):
#    Avalia TODAS as combinações possíveis de um grid predefinido.
#    Exhaustivo mas exponencialmente caro: para P parâmetros com V
#    valores cada, requer V^P avaliações.
#
# PODA (Pruning — Akiba et al., 2019):
# O Optuna pode interromper precocemente trials pouco promissores usando
# o MedianPruner, que compara o desempenho intermediário de cada trial
# com a mediana dos trials anteriores na mesma época. Se o trial está
# significativamente abaixo da mediana, ele é podado, economizando
# recursos computacionais que seriam desperdiçados em treinamentos
# condenados a falhar.
#
# Referências:
#   - Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
#     "Optuna: A Next-generation Hyperparameter Optimization Framework."
#     Proceedings of KDD.
#   - Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter
#     Optimization." JMLR 13, 281-305.
#   - Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).
#     "Algorithms for Hyper-Parameter Optimization." NeurIPS.
# ============================================================================

import os
import logging
import yaml

import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, RandomSampler, GridSampler

from src.data_loader import load_mnist
from src.architectures import build_model, count_parameters
from src.training import Trainer, build_optimizer, build_scheduler
from src.evaluation import collect_predictions

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


# ============================================================================
# FUNÇÃO OBJETIVO DO OPTUNA
# ============================================================================

def create_objective(
    data_dir: str,
    search_space: dict,
    device: torch.device,
    seed: int = 42,
    pruning_warmup: int = 5,
):
    """
    Factory function que cria a função objetivo para o Optuna.

    No Optuna, a "função objetivo" é a função que recebe um Trial
    (uma combinação de hiperparâmetros) e retorna o valor da métrica
    a ser otimizada. O Optuna chama esta função repetidamente, cada
    vez com hiperparâmetros diferentes, buscando maximizar (ou minimizar)
    o valor retornado.

    Usamos o padrão CLOSURE (função que retorna função) para "capturar"
    as variáveis de configuração (data_dir, search_space, etc.) no escopo
    da função interna, sem precisar de variáveis globais.

    Parâmetros:
        data_dir (str): Diretório dos dados MNIST.
        search_space (dict): Espaço de busca definido no YAML.
        device (torch.device): Dispositivo de computação.
        seed (int): Semente para reprodutibilidade.
        pruning_warmup (int): Épocas antes de permitir poda.

    Retorna:
        callable: Função objetivo que o Optuna irá otimizar.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Função objetivo que treina um modelo com hiperparâmetros sugeridos
        pelo Optuna e retorna a acurácia de validação.

        Fluxo interno:
        1. O Optuna sugere valores para cada hiperparâmetro.
        2. O modelo é construído com esses hiperparâmetros.
        3. O modelo é treinado e validado.
        4. A acurácia de validação é retornada como métrica de qualidade.
        5. O Optuna usa essa métrica para guiar a próxima sugestão.

        Parâmetros:
            trial (optuna.Trial): Objeto que sugere hiperparâmetros
                                  e recebe métricas intermediárias.

        Retorna:
            float: Acurácia de validação do modelo treinado.
        """
        # =================================================================
        # 1. SUGESTÃO DE HIPERPARÂMETROS
        # =================================================================
        # O Optuna oferece métodos de sugestão tipados que respeitam o
        # tipo e o intervalo de cada hiperparâmetro:
        #
        # - suggest_categorical: escolhe entre opções discretas.
        # - suggest_float: amostra um float de um intervalo contínuo.
        #   log=True → amostragem log-uniforme (adequada para lr, weight_decay).
        # - suggest_int: amostra um inteiro de um intervalo.

        # Arquitetura da rede neural.
        architecture = trial.suggest_categorical(
            'architecture', search_space['architecture']
        )

        # Taxa de aprendizado (escala logarítmica).
        lr_config = search_space['learning_rate']
        learning_rate = trial.suggest_float(
            'learning_rate',
            lr_config['low'],
            lr_config['high'],
            log=True,  # Amostragem log-uniforme: igualmente provável
        )              # entre 1e-4 e 1e-3 quanto entre 1e-2 e 1e-1.

        # Tamanho do batch.
        batch_size = trial.suggest_categorical(
            'batch_size', search_space['batch_size']
        )

        # Otimizador.
        optimizer_name = trial.suggest_categorical(
            'optimizer', search_space['optimizer']
        )

        # Número de épocas.
        epochs = trial.suggest_categorical(
            'epochs', search_space['epochs']
        )

        # Taxa de dropout.
        dr_config = search_space['dropout_rate']
        dropout_rate = trial.suggest_float(
            'dropout_rate', dr_config['low'], dr_config['high']
        )

        # Weight decay (escala logarítmica).
        wd_config = search_space['weight_decay']
        weight_decay = trial.suggest_float(
            'weight_decay', wd_config['low'], wd_config['high'], log=True
        )

        # Scheduler de learning rate.
        scheduler_name = trial.suggest_categorical(
            'scheduler', search_space['scheduler']
        )

        # Log dos hiperparâmetros sugeridos para rastreabilidade.
        logger.info(
            f"\n{'='*60}\n"
            f"Trial {trial.number}: {architecture} | LR={learning_rate:.2e} | "
            f"BS={batch_size} | Opt={optimizer_name} | "
            f"Epochs={epochs} | Drop={dropout_rate:.3f} | "
            f"WD={weight_decay:.2e} | Sched={scheduler_name}\n"
            f"{'='*60}"
        )

        # =================================================================
        # 2. CARREGAMENTO DOS DADOS
        # =================================================================
        # O DataLoader é recriado para cada trial com o batch_size sugerido.
        # Os dados são carregados do disco uma vez e cacheados pelo OS.
        dataloaders = load_mnist(
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
        )

        # =================================================================
        # 3. CONSTRUÇÃO DO MODELO
        # =================================================================
        model = build_model(architecture, dropout_rate=dropout_rate)
        num_params = count_parameters(model)
        logger.info(f"  Modelo: {architecture} ({num_params:,} parâmetros)")

        # Registra o número de parâmetros como atributo do trial.
        # Isso permite que o módulo de persistência acesse essa
        # informação via study.best_trial.user_attrs sem precisar
        # reconstruir o modelo após o tuning.
        trial.set_user_attr('num_parameters', num_params)

        # =================================================================
        # 4. CONFIGURAÇÃO DO OTIMIZADOR E SCHEDULER
        # =================================================================
        optimizer = build_optimizer(
            model, optimizer_name, learning_rate, weight_decay
        )
        scheduler = build_scheduler(optimizer, scheduler_name, epochs)

        # =================================================================
        # 5. FUNÇÃO DE PERDA
        # =================================================================
        # CrossEntropyLoss é a função de perda padrão para classificação
        # multiclasse. Ela combina LogSoftmax + Negative Log-Likelihood
        # em uma operação numericamente estável.
        criterion = nn.CrossEntropyLoss()

        # =================================================================
        # 6. TREINAMENTO COM EARLY STOPPING E PRUNING
        # =================================================================
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            patience=7,  # Early stopping: para se val_loss não melhorar
        )

        # O parâmetro trial é passado ao Trainer para integração com
        # o sistema de pruning do Optuna (ver Trainer.fit).
        history = trainer.fit(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            epochs=epochs,
            trial=trial,  # Permite que Optuna pode o trial se necessário
        )

        # =================================================================
        # 7. AVALIAÇÃO FINAL
        # =================================================================
        # Retorna a melhor acurácia de validação alcançada durante o
        # treinamento (não a última época, graças ao early stopping).
        best_val_acc = trainer.best_val_acc

        logger.info(
            f"  Trial {trial.number} concluído — "
            f"Melhor Val Acc: {best_val_acc:.4f}"
        )

        return best_val_acc

    return objective


# ============================================================================
# ORQUESTRADOR DO TUNING
# ============================================================================

def run_tuning(
    config_path: str,
    data_dir: str,
    output_dir: str,
    device: torch.device,
    n_trials: int = None,
    sampler_type: str = 'tpe',
) -> optuna.Study:
    """
    Orquestra a busca de hiperparâmetros usando o Optuna.

    Esta função:
    1. Carrega a configuração do espaço de busca do YAML.
    2. Configura o sampler (estratégia de busca) e pruner.
    3. Cria um study do Optuna e executa N trials.
    4. Salva os resultados e o relatório do melhor trial.

    STUDY: no Optuna, um Study é o objeto central que gerencia a
    otimização. Ele mantém o histórico de todos os trials, calcula
    estatísticas e coordena o sampler.

    Parâmetros:
        config_path (str): Caminho do arquivo YAML de configuração.
        data_dir (str): Diretório dos dados MNIST.
        output_dir (str): Diretório para salvar resultados.
        device (torch.device): Dispositivo de computação.
        n_trials (int): Número de trials (sobrescreve o YAML se fornecido).
        sampler_type (str): 'tpe', 'random' ou 'grid'.

    Retorna:
        optuna.Study: Study completo com histórico de todos os trials.
    """
    # -----------------------------------------------------------------
    # CARREGAMENTO DA CONFIGURAÇÃO
    # -----------------------------------------------------------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    search_space  = config['search_space']
    tuning_config = config['tuning']
    seed          = config.get('seed', 42)

    # Número de trials: argumento > YAML.
    if n_trials is None:
        n_trials = tuning_config['n_trials']

    # -----------------------------------------------------------------
    # CONFIGURAÇÃO DO SAMPLER
    # -----------------------------------------------------------------
    # O sampler determina a ESTRATÉGIA de exploração do espaço.
    if sampler_type == 'tpe':
        # TPE (Tree-structured Parzen Estimator): o sampler bayesiano
        # padrão do Optuna. Converge mais rápido que Random Search
        # ao modelar a relação entre hiperparâmetros e desempenho.
        sampler = TPESampler(
            seed=seed,
            n_startup_trials=10,  # Primeiros 10 trials são aleatórios
        )                         # para construir o modelo inicial.
    elif sampler_type == 'random':
        # Random Search: baseline simples mas eficaz.
        sampler = RandomSampler(seed=seed)
    elif sampler_type == 'grid':
        # Grid Search: avalia TODAS as combinações (cuidado com a
        # explosão combinatória).
        # Para Grid Search, precisamos de um grid discreto explícito.
        grid = {
            'architecture': search_space['architecture'],
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'batch_size': search_space['batch_size'],
            'optimizer': search_space['optimizer'],
            'epochs': search_space['epochs'],
            'dropout_rate': [0.0, 0.25, 0.5],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'scheduler': search_space['scheduler'],
        }
        sampler = GridSampler(grid, seed=seed)
        n_trials = None  # Grid Search determina automaticamente o nº de trials
    else:
        raise ValueError(f"Sampler '{sampler_type}' não suportado.")

    # -----------------------------------------------------------------
    # CONFIGURAÇÃO DO PRUNER
    # -----------------------------------------------------------------
    # O MedianPruner compara cada trial com a mediana dos trials
    # anteriores na mesma época. Se o trial está significativamente
    # abaixo da mediana, ele é interrompido precocemente.
    pruner = MedianPruner(
        n_startup_trials=5,           # Espera 5 trials antes de podar
        n_warmup_steps=tuning_config.get('pruning_warmup_epochs', 5),
        interval_steps=1,             # Verifica a cada época
    ) if tuning_config.get('pruning', True) else optuna.pruners.NopPruner()

    # -----------------------------------------------------------------
    # CRIAÇÃO DO STUDY
    # -----------------------------------------------------------------
    study = optuna.create_study(
        study_name="mnist_hyperparameter_search",
        direction="maximize",          # Maximizar a acurácia de validação
        sampler=sampler,
        pruner=pruner,
    )

    # Cria a função objetivo.
    objective = create_objective(
        data_dir=data_dir,
        search_space=search_space,
        device=device,
        seed=seed,
        pruning_warmup=tuning_config.get('pruning_warmup_epochs', 5),
    )

    # -----------------------------------------------------------------
    # EXECUÇÃO DA OTIMIZAÇÃO
    # -----------------------------------------------------------------
    logger.info(
        f"\nIniciando busca de hiperparâmetros\n"
        f"  Sampler: {sampler_type.upper()}\n"
        f"  Trials: {n_trials if n_trials else 'Grid completo'}\n"
        f"  Dispositivo: {device}\n"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # -----------------------------------------------------------------
    # RELATÓRIO DOS RESULTADOS
    # -----------------------------------------------------------------
    # Melhor trial encontrado.
    best_trial = study.best_trial

    logger.info(
        f"\n{'='*60}\n"
        f"MELHOR TRIAL: #{best_trial.number}\n"
        f"  Acurácia de Validação: {best_trial.value:.4f}\n"
        f"  Hiperparâmetros:\n"
    )
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Salva o relatório em arquivo.
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'tuning_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE OTIMIZAÇÃO DE HIPERPARÂMETROS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Sampler: {sampler_type.upper()}\n")
        f.write(f"Total de Trials: {len(study.trials)}\n")
        f.write(f"Trials Completos: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}\n")
        f.write(f"Trials Podados: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}\n\n")

        f.write("-" * 60 + "\n")
        f.write("MELHOR CONFIGURAÇÃO\n")
        f.write("-" * 60 + "\n")
        f.write(f"Trial: #{best_trial.number}\n")
        f.write(f"Acurácia de Validação: {best_trial.value:.4f}\n\n")

        f.write("Hiperparâmetros:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")

        # Tabela de todos os trials (top 10).
        f.write("\n" + "-" * 60 + "\n")
        f.write("TOP 10 TRIALS\n")
        f.write("-" * 60 + "\n")

        sorted_trials = sorted(
            study.get_trials(states=[optuna.trial.TrialState.COMPLETE]),
            key=lambda t: t.value,
            reverse=True,
        )

        for i, trial in enumerate(sorted_trials[:10]):
            f.write(
                f"\n  #{trial.number} | Val Acc: {trial.value:.4f} | "
                f"Arch: {trial.params.get('architecture', 'N/A')} | "
                f"LR: {trial.params.get('learning_rate', 0):.2e} | "
                f"Opt: {trial.params.get('optimizer', 'N/A')}\n"
            )

    logger.info(f"\nRelatório de tuning salvo em: {report_path}")

    return study
