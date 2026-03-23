# ============================================================================
# MNIST Academic Baseline — Modulo de Persistencia de Hiperparametros
# ============================================================================
# Este modulo implementa o mecanismo de persistencia dos melhores
# hiperparametros encontrados pelo tuning, permitindo que execucoes
# subsequentes de treinamento reutilizem automaticamente a melhor
# configuracao ja descoberta.
#
# O sistema opera em tres etapas:
#
# 1. SALVAMENTO: Apos o tuning, os hiperparametros do melhor trial sao
#    salvos em 'results/best_params.yaml' junto com metricas de
#    desempenho e metadados da execucao.
#
# 2. COMPARACAO: Quando um novo tuning e executado, seus resultados sao
#    comparados contra o melhor resultado salvo anteriormente. A
#    comparacao usa uma hierarquia de criterios:
#      - Criterio primario: acuracia de validacao (maior e melhor).
#      - Desempate 1: numero de parametros (menor e melhor — Navalha
#        de Occam: modelos mais simples generalizam melhor).
#      - Desempate 2: tempo de treinamento (menor e melhor).
#    O novo resultado so sobrescreve o anterior se for estritamente
#    superior segundo esses criterios.
#
# 3. HISTORICO: Todas as execucoes de tuning sao registradas em
#    'results/tuning_history.yaml', independente de serem melhores ou
#    nao. Isso permite analise retrospectiva de todos os experimentos.
#
# 4. CARREGAMENTO: O modo '--mode train' detecta automaticamente a
#    existencia de 'best_params.yaml' e utiliza os hiperparametros
#    salvos em vez dos defaults do YAML, sem intervencao manual.
#
# Referencia sobre selecao de modelos:
#   - Hastie, Tibshirani & Friedman (2009), "The Elements of Statistical
#     Learning", cap. 7 — Model Assessment and Selection.
# ============================================================================

import os
import logging
import yaml
from datetime import datetime

from src.architectures import build_model, count_parameters

logger = logging.getLogger(__name__)

# Nome dos arquivos de persistencia.
BEST_PARAMS_FILENAME   = 'best_params.yaml'
TUNING_HISTORY_FILENAME = 'tuning_history.yaml'

# Diferenca minima de acuracia para considerar uma melhoria real.
# Diferencas menores que 0.01% sao tratadas como empate, evitando
# que ruido numerico de ponto flutuante cause trocas desnecessarias.
ACCURACY_THRESHOLD = 0.0001


# ============================================================================
# CONSTRUCAO DO REGISTRO DE UM TUNING
# ============================================================================

def _build_record(best_trial, sampler_type: str, n_trials: int, study) -> dict:
    """
    Constroi o dicionario completo de um resultado de tuning.

    Este dicionario contem tres secoes:
    - hyperparameters: os 8 hiperparametros otimizados.
    - metrics: metricas de desempenho para comparacao.
    - metadata: informacoes sobre a execucao do tuning.

    Parametros:
        best_trial: Objeto FrozenTrial do Optuna com os melhores resultados.
        sampler_type (str): Tipo de sampler utilizado ('tpe', 'random', 'grid').
        n_trials (int): Numero total de trials executados.
        study: Objeto Study do Optuna com o historico completo.

    Retorna:
        dict: Registro completo do resultado do tuning.
    """
    import optuna

    params = best_trial.params

    # Calcula o numero de parametros treinaveis do modelo.
    # Usa os hiperparametros do melhor trial para instanciar o modelo
    # e contar seus parametros, sem treina-lo.
    architecture = params.get('architecture', 'LeNet5')
    dropout_rate = params.get('dropout_rate', 0.25)
    model = build_model(architecture, dropout_rate=dropout_rate)
    num_params = count_parameters(model)

    # Calcula o tempo de treinamento do melhor trial.
    # O Optuna registra automaticamente a duracao de cada trial
    # no atributo 'duration' (tipo timedelta).
    training_time = 0.0
    if best_trial.duration is not None:
        training_time = best_trial.duration.total_seconds()

    # Conta trials por estado (completos, podados, falhos).
    n_completed = len(study.get_trials(
        states=[optuna.trial.TrialState.COMPLETE]
    ))
    n_pruned = len(study.get_trials(
        states=[optuna.trial.TrialState.PRUNED]
    ))

    record = {
        'hyperparameters': {
            'architecture':  params.get('architecture', 'LeNet5'),
            'learning_rate': float(params.get('learning_rate', 1e-3)),
            'batch_size':    int(params.get('batch_size', 64)),
            'optimizer':     params.get('optimizer', 'Adam'),
            'epochs':        int(params.get('epochs', 20)),
            'dropout_rate':  float(params.get('dropout_rate', 0.25)),
            'weight_decay':  float(params.get('weight_decay', 1e-4)),
            'scheduler':     params.get('scheduler', 'CosineAnnealingLR'),
        },
        'metrics': {
            'val_accuracy':           float(best_trial.value),
            'num_parameters':         num_params,
            'training_time_seconds':  round(training_time, 1),
        },
        'metadata': {
            'sampler':       sampler_type,
            'n_trials':      n_trials,
            'n_completed':   n_completed,
            'n_pruned':      n_pruned,
            'trial_number':  best_trial.number,
            'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
    }

    return record


# ============================================================================
# SALVAMENTO E CARREGAMENTO
# ============================================================================

def save_best_params(output_dir: str, record: dict) -> str:
    """
    Salva o registro dos melhores hiperparametros em um arquivo YAML.

    O arquivo e salvo em '{output_dir}/best_params.yaml' e pode ser
    lido automaticamente pelo modo '--mode train' em execucoes futuras.

    Parametros:
        output_dir (str): Diretorio de saida.
        record (dict): Registro completo do resultado do tuning.

    Retorna:
        str: Caminho absoluto do arquivo salvo.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, BEST_PARAMS_FILENAME)

    with open(filepath, 'w') as f:
        # Cabecalho informativo no arquivo gerado.
        f.write("# ============================================\n")
        f.write("# Melhores hiperparametros encontrados pelo tuning\n")
        f.write("# Gerado automaticamente — nao editar manualmente\n")
        f.write(f"# Ultima atualizacao: {record['metadata']['timestamp']}\n")
        f.write("# ============================================\n\n")
        yaml.dump(record, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Melhores hiperparametros salvos em: {filepath}")
    return filepath


def load_best_params(output_dir: str) -> dict:
    """
    Carrega os melhores hiperparametros salvos de um tuning anterior.

    Parametros:
        output_dir (str): Diretorio onde o arquivo best_params.yaml esta.

    Retorna:
        dict ou None: O registro carregado, ou None se o arquivo nao
                      existir ou estiver corrompido.
    """
    filepath = os.path.join(output_dir, BEST_PARAMS_FILENAME)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            record = yaml.safe_load(f)

        # Validacao basica: verifica se as secoes obrigatorias existem.
        if not isinstance(record, dict):
            logger.warning(f"Arquivo {filepath} possui formato invalido. Ignorando.")
            return None
        if 'hyperparameters' not in record or 'metrics' not in record:
            logger.warning(f"Arquivo {filepath} incompleto. Ignorando.")
            return None

        return record

    except Exception as e:
        logger.warning(f"Erro ao ler {filepath}: {e}. Ignorando.")
        return None


# ============================================================================
# HISTORICO DE TUNINGS
# ============================================================================

def append_to_history(output_dir: str, record: dict, was_best: bool):
    """
    Adiciona um registro ao historico cumulativo de tunings.

    O historico e uma lista YAML contendo todos os tunings ja executados,
    independente de terem sido melhores ou nao. Cada entrada inclui um
    campo 'was_saved_as_best' indicando se aquele tuning substituiu o
    melhor resultado anterior.

    Isso permite analise retrospectiva: qual sampler tende a produzir
    melhores resultados? Quantos trials sao suficientes? Qual arquitetura
    aparece com mais frequencia entre os melhores?

    Parametros:
        output_dir (str): Diretorio de saida.
        record (dict): Registro do tuning atual.
        was_best (bool): Se este tuning se tornou o novo melhor.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, TUNING_HISTORY_FILENAME)

    # Carrega o historico existente ou inicia uma lista vazia.
    history = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, list):
                    history = loaded
        except Exception:
            pass  # Se o arquivo estiver corrompido, reinicia o historico.

    # Adiciona o campo indicando se este tuning foi salvo como melhor.
    entry = record.copy()
    entry['was_saved_as_best'] = was_best

    history.append(entry)

    with open(filepath, 'w') as f:
        f.write("# Historico de todas as execucoes de tuning\n")
        f.write(f"# Total de execucoes: {len(history)}\n\n")
        yaml.dump(history, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Historico de tuning atualizado: {filepath} ({len(history)} execucoes)")


# ============================================================================
# COMPARACAO ENTRE RESULTADOS
# ============================================================================

def _is_strictly_better(new_record: dict, existing_record: dict) -> bool:
    """
    Determina se um novo resultado de tuning e estritamente melhor que
    o resultado salvo anteriormente.

    A comparacao segue uma hierarquia de criterios:

    1. ACURACIA DE VALIDACAO (criterio primario):
       Se a diferenca for maior que ACCURACY_THRESHOLD (0.01%), o resultado
       com maior acuracia vence. Isso evita que flutuacoes de ponto
       flutuante causem trocas desnecessarias.

    2. NUMERO DE PARAMETROS (desempate — Navalha de Occam):
       Se as acuracias forem equivalentes (dentro do threshold), o modelo
       com menos parametros e preferido. Modelos mais simples tendem a
       generalizar melhor e sao mais eficientes computacionalmente.
       Referencia: Goodfellow et al. (2016), cap. 5.6 — modelo mais
       simples consistente com os dados.

    3. TEMPO DE TREINAMENTO (desempate secundario):
       Se os modelos tiverem acuracia e complexidade equivalentes,
       o mais rapido de treinar e preferido.

    Parametros:
        new_record (dict): Registro do novo tuning.
        existing_record (dict): Registro do tuning anterior (salvo).

    Retorna:
        bool: True se o novo resultado e estritamente melhor.
    """
    new_acc  = new_record['metrics']['val_accuracy']
    old_acc  = existing_record['metrics']['val_accuracy']

    # Criterio 1: acuracia (com threshold para evitar ruido).
    if new_acc > old_acc + ACCURACY_THRESHOLD:
        return True
    if old_acc > new_acc + ACCURACY_THRESHOLD:
        return False

    # Acuracias sao equivalentes — aplicar desempates.

    # Criterio 2: numero de parametros (menor e melhor).
    new_params = new_record['metrics'].get('num_parameters', float('inf'))
    old_params = existing_record['metrics'].get('num_parameters', float('inf'))
    if new_params < old_params:
        return True
    if old_params < new_params:
        return False

    # Criterio 3: tempo de treinamento (menor e melhor).
    new_time = new_record['metrics'].get('training_time_seconds', float('inf'))
    old_time = existing_record['metrics'].get('training_time_seconds', float('inf'))
    return new_time < old_time


def _format_comparison_table(existing: dict, new: dict, verdict: str) -> str:
    """
    Formata uma tabela de comparacao entre o resultado salvo e o novo
    resultado de tuning para exibicao no log.

    A tabela mostra lado a lado as metricas de ambos os resultados,
    indicando qual e superior em cada aspecto.

    Parametros:
        existing (dict): Registro do resultado salvo anteriormente.
        new (dict): Registro do novo resultado.
        verdict (str): Texto descrevendo o resultado da comparacao.

    Retorna:
        str: Tabela formatada como string multi-linha.
    """
    sep = "=" * 70

    def _winner(metric_name: str, higher_is_better: bool = True) -> str:
        """Determina qual resultado e melhor para uma metrica especifica."""
        new_val = new['metrics'].get(metric_name, 0)
        old_val = existing['metrics'].get(metric_name, 0)
        if new_val == old_val:
            return "EMPATE"
        if higher_is_better:
            return "NOVO" if new_val > old_val else "SALVO"
        else:
            return "NOVO" if new_val < old_val else "SALVO"

    lines = [
        "",
        sep,
        "COMPARACAO: Novo Tuning vs. Melhor Salvo Anteriormente",
        sep,
        "",
        f"  {'Metrica':<28} {'Salvo':>14} {'Novo':>14} {'Melhor':>10}",
        f"  {'─' * 28} {'─' * 14} {'─' * 14} {'─' * 10}",
    ]

    # Metricas de desempenho.
    old_acc = existing['metrics']['val_accuracy']
    new_acc = new['metrics']['val_accuracy']
    lines.append(
        f"  {'Val Accuracy':<28} {old_acc:>14.4f} {new_acc:>14.4f} "
        f"{_winner('val_accuracy', True):>10}"
    )

    old_params = existing['metrics'].get('num_parameters', 0)
    new_params = new['metrics'].get('num_parameters', 0)
    lines.append(
        f"  {'Num Parameters':<28} {old_params:>14,} {new_params:>14,} "
        f"{_winner('num_parameters', False):>10}"
    )

    old_time = existing['metrics'].get('training_time_seconds', 0)
    new_time = new['metrics'].get('training_time_seconds', 0)
    lines.append(
        f"  {'Training Time (s)':<28} {old_time:>14.1f} {new_time:>14.1f} "
        f"{_winner('training_time_seconds', False):>10}"
    )

    # Informacoes descritivas (sem "melhor/pior").
    lines.append(f"  {'─' * 28} {'─' * 14} {'─' * 14} {'─' * 10}")

    old_arch = existing['hyperparameters']['architecture']
    new_arch = new['hyperparameters']['architecture']
    lines.append(
        f"  {'Architecture':<28} {old_arch:>14} {new_arch:>14} {'---':>10}"
    )

    old_sampler = existing['metadata']['sampler']
    new_sampler = new['metadata']['sampler']
    lines.append(
        f"  {'Sampler':<28} {old_sampler:>14} {new_sampler:>14} {'---':>10}"
    )

    old_trials = existing['metadata']['n_trials']
    new_trials = new['metadata']['n_trials']
    lines.append(
        f"  {'Trials':<28} {old_trials:>14} {new_trials:>14} {'---':>10}"
    )

    old_ts = existing['metadata']['timestamp']
    new_ts = new['metadata']['timestamp']
    lines.append(
        f"  {'Timestamp':<28} {old_ts:>14} {new_ts:>14} {'---':>10}"
    )

    lines.append("")
    lines.append(sep)
    lines.append(f"  VEREDITO: {verdict}")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# FUNCAO PRINCIPAL DE COMPARACAO E SALVAMENTO
# ============================================================================

def compare_and_maybe_save(
    output_dir: str,
    best_trial,
    sampler_type: str,
    n_trials: int,
    study,
) -> bool:
    """
    Compara o resultado do tuning atual com o melhor resultado salvo
    anteriormente e salva o novo resultado apenas se for superior.

    Esta funcao implementa o fluxo completo:
    1. Constroi o registro do tuning atual.
    2. Carrega o melhor resultado salvo (se existir).
    3. Compara os dois usando criterios hierarquicos.
    4. Exibe uma tabela de comparacao detalhada no log.
    5. Salva o novo resultado se for melhor (ou se nao houver anterior).
    6. Registra no historico independente do resultado.

    Parametros:
        output_dir (str): Diretorio de saida.
        best_trial: FrozenTrial do Optuna com o melhor resultado.
        sampler_type (str): Tipo de sampler utilizado.
        n_trials (int): Numero total de trials executados.
        study: Objeto Study do Optuna.

    Retorna:
        bool: True se o novo resultado foi salvo como melhor.
    """
    # Constroi o registro do resultado atual.
    new_record = _build_record(best_trial, sampler_type, n_trials, study)

    # Tenta carregar o resultado salvo anteriormente.
    existing_record = load_best_params(output_dir)

    if existing_record is None:
        # Nenhum resultado anterior — salvar incondicionalmente.
        logger.info(
            "Nenhum resultado de tuning anterior encontrado. "
            "Salvando como baseline inicial."
        )
        save_best_params(output_dir, new_record)
        append_to_history(output_dir, new_record, was_best=True)
        return True

    # Resultado anterior existe — comparar.
    is_better = _is_strictly_better(new_record, existing_record)

    if is_better:
        verdict = (
            "Novo resultado e SUPERIOR ao salvo anteriormente. "
            "Atualizando best_params.yaml."
        )
    else:
        verdict = (
            "Resultado salvo anteriormente e IGUAL ou SUPERIOR. "
            "best_params.yaml mantido sem alteracao."
        )

    # Exibe a tabela de comparacao no log.
    comparison = _format_comparison_table(existing_record, new_record, verdict)
    logger.info(comparison)

    if is_better:
        save_best_params(output_dir, new_record)

    # Registra no historico independente do resultado.
    append_to_history(output_dir, new_record, was_best=is_better)

    return is_better
