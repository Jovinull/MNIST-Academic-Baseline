#!/usr/bin/env python3
# ============================================================================
# MNIST Academic Baseline — Script Principal (Entry Point)
# ============================================================================
# Este é o ponto de entrada do pipeline. Ele orquestra dois modos de
# execução mutuamente exclusivos:
#
#   1. TREINAMENTO ÚNICO (--mode train):
#      Treina um modelo com os hiperparâmetros padrão definidos no YAML.
#      Útil para validação rápida e como baseline experimental.
#
#   2. OTIMIZAÇÃO DE HIPERPARÂMETROS (--mode tune):
#      Executa a busca automatizada de hiperparâmetros usando Optuna.
#      Testa múltiplas combinações de arquitetura, learning rate,
#      otimizador, scheduler, etc.
#
# USO:
#   # Treinamento único com configuração padrão:
#   python main.py --mode train
#
#   # Tuning com 50 trials usando TPE:
#   python main.py --mode tune --n_trials 50 --sampler tpe
#
#   # Tuning com Random Search e GPU:
#   python main.py --mode tune --sampler random --device cuda
#
# Referências:
#   - Paszke et al. (2019), "PyTorch: An Imperative Style, High-Performance
#     Deep Learning Library", NeurIPS.
#   - Akiba et al. (2019), "Optuna: A Next-generation Hyperparameter
#     Optimization Framework", KDD.
# ============================================================================

import argparse      # Parsing de argumentos de linha de comando
import logging       # Sistema de logging hierárquico
import os            # Operações de sistema de arquivos
import sys           # Acesso ao sistema (exit codes)
import random        # Gerador de números aleatórios do Python
import yaml          # Parsing de arquivos YAML

import numpy as np   # Operações numéricas
import torch         # Framework de Deep Learning
import torch.nn as nn

# Importações dos módulos internos do projeto.
from src.data_loader import load_mnist
from src.architectures import build_model, count_parameters
from src.training import Trainer, build_optimizer, build_scheduler
from src.evaluation import full_evaluation
from src.tuning import run_tuning


# ============================================================================
# CONFIGURAÇÃO DE REPRODUTIBILIDADE
# ============================================================================

def set_seed(seed: int):
    """
    Fixa as sementes aleatórias de TODOS os geradores envolvidos.

    A reprodutibilidade é um pilar da ciência experimental. Em ML, a
    aleatoriedade entra em múltiplos pontos do pipeline:
      - Inicialização dos pesos (PyTorch).
      - Embaralhamento dos dados (DataLoader).
      - Dropout e data augmentation.
      - Operações em GPU (cuDNN).

    Para obter resultados EXATAMENTE reprodutíveis, precisamos fixar
    a semente de TODOS esses geradores simultaneamente.

    NOTA: torch.backends.cudnn.deterministic=True e benchmark=False
    garantem determinismo na GPU, mas podem reduzir a performance em
    10-20%. Em produção, esses flags são tipicamente desativados.

    Referência: Bouthillier et al. (2021), "Accounting for Variance in
    Machine Learning Benchmarks", MLSys.

    Parâmetros:
        seed (int): Semente aleatória (valor inteiro não-negativo).
    """
    # Fixa a semente do módulo random do Python (usado em data augmentation).
    random.seed(seed)

    # Fixa a semente do NumPy (usado em operações de array e shuffling).
    np.random.seed(seed)

    # Fixa a semente do PyTorch para CPU.
    torch.manual_seed(seed)

    # Fixa a semente do PyTorch para TODAS as GPUs (se disponíveis).
    torch.cuda.manual_seed_all(seed)

    # Garante que o cuDNN use algoritmos determinísticos.
    # cuDNN normalmente seleciona o algoritmo mais RÁPIDO para cada
    # operação (e.g., convolução), que pode variar entre execuções.
    # deterministic=True força sempre o mesmo algoritmo.
    torch.backends.cudnn.deterministic = True

    # Desabilita o benchmark automático de algoritmos do cuDNN.
    # Quando True, o cuDNN testa vários algoritmos no início e escolhe
    # o mais rápido para o hardware atual. Útil em produção, mas
    # introduz não-determinismo.
    torch.backends.cudnn.benchmark = False


# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

def setup_logging(log_dir: str, log_level: str = 'INFO'):
    """
    Configura o sistema de logging para console e arquivo.

    O logging é essencial para:
    1. Monitorar o progresso do treinamento em tempo real.
    2. Diagnosticar problemas sem precisar re-executar experimentos.
    3. Manter um registro permanente dos resultados para reprodutibilidade.

    Usamos dois handlers:
      - StreamHandler: exibe logs no terminal (stdout).
      - FileHandler: salva logs em arquivo para referência futura.

    O formato inclui timestamp, módulo de origem e nível de severidade,
    seguindo as boas práticas de engenharia de software.

    Parâmetros:
        log_dir (str): Diretório para salvar o arquivo de log.
        log_level (str): Nível mínimo de severidade ('DEBUG', 'INFO', etc.).
    """
    os.makedirs(log_dir, exist_ok=True)

    # Formato das mensagens de log.
    # Exemplo: "2024-01-15 14:30:45 | INFO | src.training | Época 1/20 ..."
    log_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Configura o logger raiz (root logger).
    # Todos os loggers criados com logging.getLogger(__name__) herdam
    # esta configuração automaticamente.
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Handler para o terminal.
            logging.StreamHandler(sys.stdout),
            # Handler para arquivo persistente.
            logging.FileHandler(
                os.path.join(log_dir, 'experiment.log'),
                mode='a',  # 'a' = append (não sobrescreve logs anteriores)
            ),
        ],
    )

    # Suprime logs excessivos de bibliotecas externas.
    # Optuna e matplotlib geram muitos logs de nível DEBUG que poluem
    # a saída. Elevamos o nível mínimo deles para WARNING.
    logging.getLogger('optuna').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


# ============================================================================
# PARSING DE ARGUMENTOS DE LINHA DE COMANDO
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Define e processa os argumentos de linha de comando.

    O argparse permite que o usuário configure o pipeline sem modificar
    o código-fonte, seguindo o princípio de "Configuration over Code".

    Retorna:
        argparse.Namespace: Objeto com todos os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description='MNIST Academic Baseline — Pipeline de Classificação',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --mode train
  python main.py --mode train --architecture DeepCNN --epochs 30
  python main.py --mode tune --n_trials 50 --sampler tpe
  python main.py --mode tune --sampler random --device cuda
        """,
    )

    # --- Modo de execução ---
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'tune'],
        default='train',
        help='Modo de execução: "train" para treino único, '
             '"tune" para otimização de hiperparâmetros. (default: train)',
    )

    # --- Caminhos ---
    parser.add_argument(
        '--config',
        type=str,
        default='config/hyperparameters.yaml',
        help='Caminho do arquivo de configuração YAML. (default: config/hyperparameters.yaml)',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Diretório contendo os arquivos binários IDX do MNIST. (default: data/raw)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Diretório para salvar resultados, figuras e checkpoints. (default: results)',
    )

    # --- Dispositivo ---
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Dispositivo: "cpu", "cuda", "mps" ou "auto" (detecção automática). '
             '(default: auto)',
    )

    # --- Hiperparâmetros (override para modo train) ---
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['MLP', 'LeNet5', 'ModernCNN', 'DeepCNN'],
        default=None,
        help='Arquitetura do modelo (sobrescreve o YAML). '
             'Opções: MLP, LeNet5, ModernCNN, DeepCNN.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Número de épocas de treinamento (sobrescreve o YAML).',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Taxa de aprendizado (sobrescreve o YAML).',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Tamanho do mini-batch (sobrescreve o YAML).',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['SGD', 'Adam', 'AdamW'],
        default=None,
        help='Algoritmo de otimização (sobrescreve o YAML).',
    )

    # --- Configuração do tuning ---
    parser.add_argument(
        '--n_trials',
        type=int,
        default=None,
        help='Número de trials para otimização de hiperparâmetros.',
    )
    parser.add_argument(
        '--sampler',
        type=str,
        choices=['tpe', 'random', 'grid'],
        default='tpe',
        help='Estratégia de amostragem: "tpe" (bayesiana), "random" ou "grid". '
             '(default: tpe)',
    )

    return parser.parse_args()


# ============================================================================
# DETECÇÃO AUTOMÁTICA DE DISPOSITIVO
# ============================================================================

def get_device(device_str: str) -> torch.device:
    """
    Detecta e retorna o melhor dispositivo de computação disponível.

    Ordem de preferência:
    1. CUDA (GPU NVIDIA): aceleração massiva via paralelismo SIMT.
    2. MPS (Apple Silicon): aceleração via Metal Performance Shaders.
    3. CPU: fallback universal, mais lento mas sempre disponível.

    Parâmetros:
        device_str (str): 'auto', 'cpu', 'cuda' ou 'mps'.

    Retorna:
        torch.device: Dispositivo selecionado.
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Loga informações da GPU para diagnóstico.
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logging.info(f"GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info("Apple MPS detectado")
        else:
            device = torch.device('cpu')
            logging.info("Nenhuma GPU detectada, usando CPU")
    else:
        device = torch.device(device_str)

    return device


# ============================================================================
# MODO: TREINAMENTO ÚNICO
# ============================================================================

def run_single_training(args: argparse.Namespace, config: dict, device: torch.device):
    """
    Executa um treinamento único com os hiperparâmetros padrão ou
    sobrescritos via linha de comando.

    Fluxo:
    1. Resolve hiperparâmetros (CLI override > YAML defaults).
    2. Carrega dados e cria DataLoaders.
    3. Constrói modelo, otimizador, scheduler e função de perda.
    4. Treina o modelo com early stopping.
    5. Avalia no conjunto de teste e gera relatórios.

    Parâmetros:
        args (Namespace): Argumentos de linha de comando.
        config (dict): Configuração carregada do YAML.
        device (torch.device): Dispositivo de computação.
    """
    logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------
    # RESOLUÇÃO DE HIPERPARÂMETROS
    # -----------------------------------------------------------------
    # Padrão: usa os valores do YAML. Se o usuário passou argumentos
    # pela CLI, eles sobrescrevem os defaults.
    defaults = config['defaults']

    architecture  = args.architecture or defaults['architecture']
    epochs        = args.epochs or defaults['epochs']
    learning_rate = args.learning_rate or defaults['learning_rate']
    batch_size    = args.batch_size or defaults['batch_size']
    optimizer_name = args.optimizer or defaults['optimizer']
    dropout_rate  = defaults['dropout_rate']
    weight_decay  = defaults['weight_decay']
    scheduler_name = defaults['scheduler']
    seed          = config.get('seed', 42)

    logger.info(
        f"\n{'='*60}\n"
        f"TREINAMENTO ÚNICO\n"
        f"{'='*60}\n"
        f"  Arquitetura:    {architecture}\n"
        f"  Épocas:         {epochs}\n"
        f"  Learning Rate:  {learning_rate}\n"
        f"  Batch Size:     {batch_size}\n"
        f"  Otimizador:     {optimizer_name}\n"
        f"  Dropout:        {dropout_rate}\n"
        f"  Weight Decay:   {weight_decay}\n"
        f"  Scheduler:      {scheduler_name}\n"
        f"  Seed:           {seed}\n"
        f"  Dispositivo:    {device}\n"
        f"{'='*60}"
    )

    # -----------------------------------------------------------------
    # 1. CARREGAMENTO DOS DADOS
    # -----------------------------------------------------------------
    data_config = config.get('data', {})
    dataloaders = load_mnist(
        data_dir=args.data_dir,
        batch_size=batch_size,
        validation_split=data_config.get('validation_split', 0.1),
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', True),
        seed=seed,
    )

    logger.info("Dados carregados com sucesso.")

    # -----------------------------------------------------------------
    # 2. CONSTRUÇÃO DO MODELO
    # -----------------------------------------------------------------
    model = build_model(architecture, dropout_rate=dropout_rate)
    num_params = count_parameters(model)
    logger.info(f"Modelo: {architecture} ({num_params:,} parâmetros treináveis)")

    # Exibe a arquitetura completa do modelo no log.
    # repr(model) mostra todas as camadas, shapes e hiperparâmetros.
    logger.info(f"\nArquitetura do modelo:\n{model}")

    # -----------------------------------------------------------------
    # 3. OTIMIZADOR, SCHEDULER E FUNÇÃO DE PERDA
    # -----------------------------------------------------------------
    optimizer = build_optimizer(
        model, optimizer_name, learning_rate, weight_decay
    )
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------
    # 4. TREINAMENTO
    # -----------------------------------------------------------------
    checkpoint_path = os.path.join(
        args.output_dir, 'checkpoints', f'{architecture}_best.pt'
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        patience=7,
        checkpoint_path=checkpoint_path,
    )

    history = trainer.fit(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=epochs,
    )

    logger.info("Treinamento concluído.")

    # -----------------------------------------------------------------
    # 5. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
    # -----------------------------------------------------------------
    results = full_evaluation(
        model=model,
        test_loader=dataloaders['test'],
        device=device,
        history=history,
        output_dir=args.output_dir,
        experiment_name=architecture,
    )

    logger.info(
        f"\n{'='*60}\n"
        f"RESULTADO FINAL\n"
        f"{'='*60}\n"
        f"  Arquitetura:         {architecture}\n"
        f"  Acurácia de Teste:   {results['test_accuracy']:.4f} "
        f"({results['test_accuracy']*100:.2f}%)\n"
        f"{'='*60}"
    )


# ============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """
    Função principal que orquestra todo o pipeline.

    Sequência de inicialização:
    1. Parsear argumentos de linha de comando.
    2. Carregar configuração YAML.
    3. Configurar logging, seed e dispositivo.
    4. Despachar para o modo selecionado (train ou tune).
    """
    # 1. Parsear argumentos.
    args = parse_args()

    # 2. Carregar configuração YAML.
    if not os.path.exists(args.config):
        print(f"ERRO: Arquivo de configuração não encontrado: {args.config}")
        print("Execute a partir do diretório raiz do projeto.")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Configurar ambiente.
    seed = config.get('seed', 42)
    set_seed(seed)
    setup_logging(os.path.join(args.output_dir, 'logs'))
    device = get_device(args.device)

    logger = logging.getLogger(__name__)
    logger.info(f"Semente aleatória fixada: {seed}")
    logger.info(f"Dispositivo selecionado: {device}")

    # 4. Despachar para o modo selecionado.
    if args.mode == 'train':
        run_single_training(args, config, device)
    elif args.mode == 'tune':
        study = run_tuning(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=device,
            n_trials=args.n_trials,
            sampler_type=args.sampler,
        )

        # Após o tuning, treina o melhor modelo com avaliação completa.
        logger.info("\nTreinando o melhor modelo encontrado pelo tuning...")

        best_params = study.best_trial.params
        # Injeta os melhores hiperparâmetros como argumentos CLI.
        args.architecture  = best_params.get('architecture', 'LeNet5')
        args.epochs        = best_params.get('epochs', 20)
        args.learning_rate = best_params.get('learning_rate', 1e-3)
        args.batch_size    = best_params.get('batch_size', 64)
        args.optimizer     = best_params.get('optimizer', 'Adam')

        # Atualiza os defaults do config com os melhores hiperparâmetros.
        config['defaults'].update({
            'architecture':  args.architecture,
            'epochs':        args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size':    args.batch_size,
            'optimizer':     args.optimizer,
            'dropout_rate':  best_params.get('dropout_rate', 0.25),
            'weight_decay':  best_params.get('weight_decay', 1e-4),
            'scheduler':     best_params.get('scheduler', 'CosineAnnealingLR'),
        })

        run_single_training(args, config, device)

    logger.info("Pipeline concluído com sucesso.")


# Ponto de entrada: executa main() apenas quando o script é chamado
# diretamente (não quando importado como módulo).
if __name__ == '__main__':
    main()
