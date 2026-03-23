# ============================================================================
# MNIST Academic Baseline — Módulo de Avaliação e Visualização
# ============================================================================
# Este módulo implementa a avaliação final do modelo treinado e a geração
# de visualizações científicas. Uma avaliação rigorosa é essencial para
# validar as conclusões experimentais e comunicar resultados de forma
# clara e reprodutível.
#
# Métricas implementadas:
#   - Acurácia (accuracy): fração de predições corretas.
#   - Precisão (precision): de todas as predições positivas para uma classe,
#     quantas estão corretas. Penaliza falsos positivos.
#   - Recall (sensibilidade): de todos os exemplos reais de uma classe,
#     quantos foram detectados. Penaliza falsos negativos.
#   - F1-Score: média harmônica de precisão e recall, balanceando ambos.
#   - Matriz de Confusão: tabela NxN mostrando predições vs. verdade.
#
# Visualizações:
#   - Curvas de aprendizado (learning curves): loss e accuracy por época.
#   - Matriz de confusão com heatmap.
#   - Exemplos de erros de classificação (análise qualitativa).
#
# Referências:
#   - Sokolova & Lapalme (2009), "A systematic analysis of performance
#     measures for classification tasks", Information Processing & Management.
#   - Powers (2011), "Evaluation: From Precision, Recall and F-Measure
#     to ROC, Informedness, Markedness & Correlation", Journal of Machine
#     Learning Technologies.
# ============================================================================

import os
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para servidores sem display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,  # Relatório completo por classe
    confusion_matrix,       # Matriz de confusão NxN
    accuracy_score,         # Acurácia global
)

logger = logging.getLogger(__name__)

# Nomes das classes do MNIST (dígitos 0-9).
CLASS_NAMES = [str(i) for i in range(10)]


# ============================================================================
# COLETA DE PREDIÇÕES
# ============================================================================

def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Coleta todas as predições e rótulos verdadeiros do modelo sobre um
    dataset completo.

    Este é o primeiro passo da avaliação: precisamos de TODAS as predições
    para calcular as métricas agregadas. Diferente do treinamento, onde
    processamos batch a batch e descartamos, aqui armazenamos tudo.

    Parâmetros:
        model (nn.Module): Modelo treinado em modo eval().
        dataloader (DataLoader): DataLoader do conjunto de teste.
        device (torch.device): Dispositivo de computação.

    Retorna:
        tuple: (all_labels, all_predictions, all_probabilities)
            - all_labels: np.array de shape (N,) — rótulos verdadeiros.
            - all_predictions: np.array de shape (N,) — classes preditas.
            - all_probabilities: np.array de shape (N, 10) — probabilidades
              softmax para cada classe.
    """
    model.eval()

    all_labels   = []
    all_preds    = []
    all_probs    = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)

            # Forward pass: obtém logits brutos.
            outputs = model(images)

            # Converte logits em probabilidades via Softmax.
            # Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
            # Isso normaliza os logits para que somem 1, permitindo
            # interpretação probabilística da confiança do modelo.
            probabilities = torch.softmax(outputs, dim=1)

            # Classe predita = argmax das probabilidades.
            _, predicted = torch.max(outputs, dim=1)

            # Move para CPU e converte para NumPy para uso com sklearn.
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())

    # Concatena todos os batches em arrays contíguos.
    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)

    return all_labels, all_preds, all_probs


# ============================================================================
# RELATÓRIO DE MÉTRICAS
# ============================================================================

def generate_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> str:
    """
    Gera um relatório completo de métricas por classe.

    O relatório inclui, para cada classe (dígito 0-9):
      - Precisão: TP / (TP + FP) — "das predições positivas, quantas corretas?"
      - Recall:   TP / (TP + FN) — "dos exemplos reais, quantos detectados?"
      - F1-Score: 2 × (P × R) / (P + R) — média harmônica de P e R.
      - Support:  número de exemplos reais de cada classe no conjunto.

    Também inclui médias macro (média simples entre classes), weighted
    (ponderada pelo support) e global accuracy.

    A média MACRO trata todas as classes igualmente, independente do número
    de exemplos. A média WEIGHTED pondera pelo número de exemplos, refletindo
    melhor o desempenho em datasets desbalanceados (embora o MNIST seja
    razoavelmente balanceado).

    Parâmetros:
        labels (np.ndarray): Rótulos verdadeiros.
        predictions (np.ndarray): Predições do modelo.

    Retorna:
        str: Relatório formatado como tabela de texto.
    """
    # sklearn.metrics.classification_report gera o relatório completo.
    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,  # 4 casas decimais para precisão científica
    )

    # Calcula a acurácia global separadamente para destaque.
    accuracy = accuracy_score(labels, predictions)

    logger.info(f"\nAcurácia Global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"\nRelatório de Classificação:\n{report}")

    return report


# ============================================================================
# VISUALIZAÇÕES CIENTÍFICAS
# ============================================================================

def plot_learning_curves(history: dict, save_path: str):
    """
    Plota as curvas de aprendizado (learning curves) do treinamento.

    As learning curves são o diagnóstico mais importante para entender
    o comportamento do modelo durante o treinamento:

    1. PERDA (Loss) — Treino vs. Validação:
       - Se ambas diminuem: o modelo está aprendendo.
       - Se treino diminui mas validação estagna/aumenta: OVERFITTING.
         O modelo memoriza o treino mas não generaliza.
       - Se ambas estão altas: UNDERFITTING. O modelo é muito simples
         ou precisa de mais treinamento.

    2. ACURÁCIA — Treino vs. Validação:
       - O gap entre treino e validação mede o grau de overfitting.
       - Uma acurácia de treino muito superior à de validação indica
         que o modelo está "decorando" os dados de treino.

    Referência: Goodfellow et al. (2016), cap. 5, Fig. 5.3 — "Typical
    learning curves showing training and test set error as a function
    of training time."

    Parâmetros:
        history (dict): Dicionário com listas de métricas por época.
        save_path (str): Caminho para salvar a figura.
    """
    # Cria uma figura com 2 subplots lado a lado.
    # figsize=(14, 5) garante boa legibilidade em relatórios.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # ----- SUBPLOT 1: CURVA DE PERDA -----
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-o',
             label='Treino', markersize=3, linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-s',
             label='Validação', markersize=3, linewidth=1.5)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Perda (Loss)', fontsize=12)
    ax1.set_title('Curva de Perda', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)  # Grid sutil para leitura de valores

    # ----- SUBPLOT 2: CURVA DE ACURÁCIA -----
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-o',
             label='Treino', markersize=3, linewidth=1.5)
    ax2.plot(epochs, history['val_acc'], 'r-s',
             label='Validação', markersize=3, linewidth=1.5)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Acurácia', fontsize=12)
    ax2.set_title('Curva de Acurácia', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Ajuste automático de layout para evitar sobreposição de textos.
    plt.tight_layout()

    # Salva a figura em alta resolução para publicação.
    # dpi=150 é adequado para relatórios e apresentações.
    # bbox_inches='tight' remove margens excessivas.
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Libera memória (importante em loops de tuning)

    logger.info(f"Curvas de aprendizado salvas em: {save_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    normalize: bool = True,
):
    """
    Plota a matriz de confusão como um heatmap anotado.

    A MATRIZ DE CONFUSÃO é uma tabela C×C (C = número de classes) onde:
      - Linha i: exemplos da classe real i.
      - Coluna j: predições para a classe j.
      - Célula (i, j): número de exemplos da classe i preditos como j.

    A diagonal principal contém os ACERTOS (True Positives de cada classe).
    Células fora da diagonal são ERROS, e padrões nesses erros revelam
    confusões sistemáticas do modelo (e.g., '4' confundido com '9').

    Quando normalizada por linha (normalize=True), cada célula mostra a
    PROPORÇÃO dos exemplos de cada classe, permitindo comparação justa
    entre classes com diferentes quantidades de exemplos.

    Parâmetros:
        labels (np.ndarray): Rótulos verdadeiros.
        predictions (np.ndarray): Predições do modelo.
        save_path (str): Caminho para salvar a figura.
        normalize (bool): Se True, normaliza por classe (linha).
    """
    # Calcula a matriz de confusão usando scikit-learn.
    cm = confusion_matrix(labels, predictions)

    if normalize:
        # Normalização por linha: divide cada linha pela soma da linha.
        # Isso transforma contagens absolutas em proporções [0, 1].
        # .astype('float') evita divisão inteira (Python 3 faz float por
        # padrão, mas o NumPy não necessariamente).
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'      # Formato de 2 casas decimais para proporções
        title = 'Matriz de Confusão (Normalizada)'
    else:
        cm_normalized = cm
        fmt = 'd'        # Formato inteiro para contagens
        title = 'Matriz de Confusão'

    # Cria o heatmap usando Seaborn, que produz visualizações
    # cientificamente precisas com paleta de cores perceptualmente uniforme.
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,              # Mostra valores nas células
        fmt=fmt,                 # Formato numérico
        cmap='Blues',            # Paleta de cores (azul = mais intenso)
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,             # Células quadradas
        linewidths=0.5,          # Linhas entre células
        ax=ax,
    )

    ax.set_xlabel('Classe Predita', fontsize=12)
    ax.set_ylabel('Classe Verdadeira', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Matriz de confusão salva em: {save_path}")


def plot_misclassified_examples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    num_examples: int = 25,
):
    """
    Visualiza exemplos incorretamente classificados pelo modelo.

    A análise qualitativa dos erros é tão importante quanto as métricas
    quantitativas. Ao examinar QUAIS exemplos o modelo erra, podemos:
    1. Identificar padrões de confusão (e.g., '3' vs '8', '4' vs '9').
    2. Detectar exemplos ruidosos ou ambíguos no dataset.
    3. Avaliar se os erros são "razoáveis" (confusões humanas) ou
       "absurdos" (indicando problemas no modelo).

    Parâmetros:
        model (nn.Module): Modelo treinado.
        dataloader (DataLoader): DataLoader do conjunto de teste.
        device (torch.device): Dispositivo de computação.
        save_path (str): Caminho para salvar a figura.
        num_examples (int): Número de erros a visualizar (máximo 25).
    """
    model.eval()

    misclassified_images = []
    misclassified_labels = []
    misclassified_preds  = []

    with torch.no_grad():
        for images, labels in dataloader:
            images_dev = images.to(device, non_blocking=True)
            outputs = model(images_dev)
            _, predicted = torch.max(outputs, dim=1)

            # Identifica amostras incorretamente classificadas.
            # O operador != retorna um tensor booleano.
            mask = predicted.cpu() != labels
            if mask.any():
                # Seleciona apenas as amostras erradas deste batch.
                misclassified_images.extend(images[mask])
                misclassified_labels.extend(labels[mask].numpy())
                misclassified_preds.extend(predicted.cpu()[mask].numpy())

            # Para quando temos exemplos suficientes.
            if len(misclassified_images) >= num_examples:
                break

    # Limita ao número solicitado.
    num_show = min(num_examples, len(misclassified_images))
    if num_show == 0:
        logger.info("Nenhum exemplo incorretamente classificado encontrado!")
        return

    # Cria uma grade de subplots para exibir os erros.
    cols = 5
    rows = (num_show + cols - 1) // cols  # Arredondamento para cima
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    # Garante que axes seja sempre um array 2D para indexação uniforme.
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < num_show:
            # Exibe a imagem em escala de cinza.
            # squeeze() remove a dimensão de canal: (1, 28, 28) → (28, 28).
            img = misclassified_images[idx].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            true_label = misclassified_labels[idx]
            pred_label = misclassified_preds[idx]
            ax.set_title(
                f"Real: {true_label} | Pred: {pred_label}",
                fontsize=9,
                color='red',  # Vermelho para indicar erro
            )
        ax.axis('off')  # Remove eixos para visualização limpa

    plt.suptitle(
        'Exemplos Incorretamente Classificados',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Exemplos de erros salvos em: {save_path}")


# ============================================================================
# AVALIAÇÃO COMPLETA (FUNÇÃO ORQUESTRADORA)
# ============================================================================

def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    history: dict,
    output_dir: str,
    experiment_name: str = "experiment",
) -> dict:
    """
    Executa a avaliação completa do modelo e gera todos os artefatos.

    Esta função orquestra:
    1. Coleta de predições sobre o conjunto de teste.
    2. Geração do relatório de métricas.
    3. Plotagem das curvas de aprendizado.
    4. Plotagem da matriz de confusão.
    5. Visualização de exemplos incorretamente classificados.

    Todos os artefatos são salvos em disco para inclusão em relatórios
    e publicações.

    Parâmetros:
        model (nn.Module): Modelo treinado.
        test_loader (DataLoader): DataLoader do conjunto de teste.
        device (torch.device): Dispositivo de computação.
        history (dict): Histórico de métricas do treinamento.
        output_dir (str): Diretório para salvar os artefatos.
        experiment_name (str): Nome do experimento (prefixo dos arquivos).

    Retorna:
        dict: Dicionário com acurácia de teste e relatório de classificação.
    """
    # Garante que o diretório de saída exista.
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Coleta predições sobre o teste.
    labels, predictions, probabilities = collect_predictions(
        model, test_loader, device
    )

    # 2. Gera relatório de métricas.
    report = generate_classification_report(labels, predictions)
    test_accuracy = accuracy_score(labels, predictions)

    # 3. Curvas de aprendizado.
    plot_learning_curves(
        history,
        os.path.join(figures_dir, f'{experiment_name}_learning_curves.png'),
    )

    # 4. Matriz de confusão.
    plot_confusion_matrix(
        labels, predictions,
        os.path.join(figures_dir, f'{experiment_name}_confusion_matrix.png'),
    )

    # 5. Exemplos de erros.
    plot_misclassified_examples(
        model, test_loader, device,
        os.path.join(figures_dir, f'{experiment_name}_misclassified.png'),
    )

    # Salva o relatório em texto.
    report_path = os.path.join(output_dir, f'{experiment_name}_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Acurácia de Teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
        f.write(report)

    logger.info(f"Relatório salvo em: {report_path}")

    return {
        'test_accuracy': test_accuracy,
        'report': report,
    }
