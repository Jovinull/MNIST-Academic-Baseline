# ============================================================================
# MNIST Academic Baseline — Pacote Principal (src)
# ============================================================================
# Este arquivo __init__.py marca o diretório 'src' como um pacote Python,
# permitindo importações relativas entre os módulos do projeto.
#
# A modularização do código em subpacotes segue o princípio de Separação de
# Responsabilidades (Separation of Concerns — Dijkstra, 1974), onde cada
# módulo encapsula uma etapa distinta do pipeline de Machine Learning:
#
#   data_loader.py    → Leitura e parsing dos arquivos binários IDX
#   preprocessing.py  → Normalização e transformações dos dados
#   architectures.py  → Definição das arquiteturas de redes neurais
#   training.py       → Loop de treinamento e validação
#   evaluation.py     → Métricas de desempenho e visualizações
#   tuning.py         → Otimização de hiperparâmetros com Optuna
#   persistence.py    → Persistência e comparação de resultados de tuning
# ============================================================================
