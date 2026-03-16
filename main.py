import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Importacoes dos modulos internos do projeto
from src.cleaning import (
    read_data, 
    handle_outliers,
    clean_and_impute_data, 
    group_neighborhoods_by_price,
    add_feature_engineering,
    get_mutual_info_scores, 
    preparar_dados_para_xgboost
)
from src.utils import (
    plot_distributions, 
    plot_informacao_mutua, 
    plot_correlation_heatmap, 
    plot_feature_importance, 
    plot_regression_results,
    plot_feature_relationships,
    analyze_regression_error_by_segment
)
from src.models import (
    treinar_e_avaliar_xgboost, 
    gerar_submissao
)
from src.mining import (
    discretize_numeric, 
    dataframe_to_transactions, 
    mine_association_rules,
    mine_sequential_patterns,
    perform_clustering,
    mine_neighborhood_evolution
)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def print_separator(title):
    print("\n" + "-"*60)
    print(f" {title.upper()} ".center(60, "-"))
    print("-"*60)

def main():
    # 1. Configuracoes de Caminhos e Ambiente
    TRAIN_PATH = "data/raw/train.csv"
    TEST_PATH = "data/raw/test.csv"
    PROCESSED_DIR = "data/processed"
    FIGURES_DIR = "reports/figures"
    REPORTS_DIR = "reports"
    DIST_DIR = os.path.join(FIGURES_DIR, "distributions")
    SUBMISSION_PATH = os.path.join(PROCESSED_DIR, "submission.csv")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DIST_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    start_time = datetime.now()
    log_filename = os.path.join(REPORTS_DIR, f"execution_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout # Redireciona erros para o mesmo log

    print_separator("Iniciando Pipeline de House Price Discovery")
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log de execucao salvo em: {log_filename}")

    # 2. Carregamento de Dados
    print_separator("Fase 1: Carregamento de Dados")
    train_df = read_data(TRAIN_PATH)
    if train_df.empty:
        print("Erro critico: Dados de treino nao encontrados.")
        return

    train_df = handle_outliers(train_df, is_train=True)

    print(f"\nESTATISTICAS DO ALVO (SalePrice):")
    print(train_df['SalePrice'].describe().to_string())

    # 3. Limpeza, Imputacao e Feature Engineering
    print_separator("Fase 2: Preparacao de Dados")
    
    print("\nESTRATEGIA DE LIMPEZA E IMPUTACAO:")
    print(" - Categoricos: 'NA' para ausencia da caracteristica.")
    print(" - GarageYrBlt: 0 se sem garagem, ou YearBuilt se possui.")
    print(" - LotFrontage: Mediana do Bairro.")
    print(" - Fallback: Mediana para numericos e Moda para categoricos.")

    train_cleaned = clean_and_impute_data(train_df)
    train_nb, nb_mapping = group_neighborhoods_by_price(train_cleaned, bins=5)
    train_feat = add_feature_engineering(train_nb)
    
    # 4. Analise Exploratoria (EDA) e Correlacoes
    print_separator("Fase 3: Analise Exploratoria e Correlacoes")
    
    print("\nCalculando Informacao Mutua (MI)...")
    mi_scores = get_mutual_info_scores(train_feat)
    print(f"Top 30 atributos por MI:")
    print(mi_scores.head(30).to_string())
    
    print("\nGerando e salvando visualizacoes...")
    plot_informacao_mutua(mi_scores, top_n=30, save_path=os.path.join(FIGURES_DIR, "informacao_mutua.png"), show=False)
    plot_correlation_heatmap(train_feat, target_column='SalePrice', top_n=15, save_path=os.path.join(FIGURES_DIR, "heatmap_correlacao.png"), show=False)
    
    top_3_num = mi_scores.drop(['SalePrice'], errors='ignore').head(3).index.tolist()
    plot_feature_relationships(train_feat, 'SalePrice', top_3_num, save_dir=FIGURES_DIR, show=False)
    
    # Selecao de colunas para distribuicao (Top 10 MI + Solicitacoes do Usuario)
    top_10_features = mi_scores.head(10).index.tolist()
    requested_cols = ['BsmtQual', 'KitchenQual', '1stFlrSF', 'GarageCars', 'TotalBathrooms', 'CentralAir', 'Fireplaces', 'OverallCond']
    all_dist_cols = list(dict.fromkeys(top_10_features + requested_cols)) # Mantem ordem e remove duplicatas
    
    # Filtrar apenas as colunas que existem no dataframe
    all_dist_cols = [c for c in all_dist_cols if c in train_feat.columns]
    
    num_cols = train_feat[all_dist_cols].select_dtypes(include=np.number).columns.tolist()
    cat_cols = train_feat[all_dist_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Gerando plots de distribuicao em {DIST_DIR}...")
    plot_distributions(train_feat, numerical_cols=num_cols, categorical_cols=cat_cols, save_dir=DIST_DIR, show=False)

    # 5. Mineracao de Dados
    print_separator("Fase 4: Mineracao de Padroes Frequentes")
    
    train_clust = perform_clustering(train_feat, n_clusters=5)
    df_disc = discretize_numeric(train_clust, bins=5)
    
    print("Minerando Regras de Associacao (Apriori)...")
    relevant_cols_mining = mi_scores.head(20).index.tolist() + ['SalePriceBin', 'MarketSegment', 'NbPriceTier']
    transactions = dataframe_to_transactions(df_disc[relevant_cols_mining])
    freq, rules, rules_target = mine_association_rules(transactions, min_sup=0.08, min_conf=0.75)
    
    if not rules_target.empty:
        print(f"{len(rules_target)} Regras encontradas com o alvo.")
        print("\nTOP 20 REGRAS DE ASSOCIACAO (Ordenadas por Lift):")
        print(rules_target[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
              .sort_values('lift', ascending=False).head(20).to_string(index=False))

    print("\nMinerando Padroes Sequenciais (PrefixSpan)...")
    top_10_mi_names = sorted(mi_scores.head(10).index.tolist() + ['MarketSegment', 'NbPriceTier'])
    sequences = df_disc[top_10_mi_names].astype(str).values.tolist()
    seq_patterns = mine_sequential_patterns(sequences, min_sup_ratio=0.1)
    
    print(f"{len(seq_patterns)} Padroes de atributos encontrados.")
    for supp, pat in seq_patterns[:10]:
        print(f"   Suporte: {supp:>4} | Padrao: {pat}")

    df_disc['YearBuilt_Num'] = train_feat['YearBuilt'] 
    evol_patterns = mine_neighborhood_evolution(df_disc, min_sup_ratio=0.2)
    print(f"{len(evol_patterns)} Padroes de evolucao encontrados.")
    for supp, pat in evol_patterns[:10]:
        print(f"   Suporte: {supp:>4} | Evolucao: {pat}")

    # 6. Treinamento e Avaliacao do Modelo
    print_separator("Fase 5: Modelagem Preditiva (XGBoost)")
    
    print("\nPreparando dados para o modelo...")
    train_prep = preparar_dados_para_xgboost(train_feat)
    
    print("\nIniciando treinamento com Validacao Cruzada (10-fold)...")
    modelo, train_columns = treinar_e_avaliar_xgboost(train_prep)
    
    print("\nGerando graficos de resultados e residuos...")
    X_final = train_prep.drop(columns=['Id', 'SalePrice'], errors='ignore')
    y_final_log = np.log1p(train_prep['SalePrice'])
    y_pred_log = modelo.predict(X_final)
    
    plot_regression_results(y_final_log, y_pred_log, title="XGBoost Final", 
                           save_path=os.path.join(FIGURES_DIR, "resultados_regressao.png"), show=False)

    importances = modelo.feature_importances_
    feat_imp_df = pd.DataFrame({'Importance': importances}, index=X_final.columns).sort_values(by='Importance', ascending=False)
    plot_feature_importance(feat_imp_df, top_n=20, save_path=os.path.join(FIGURES_DIR, "importancia_features_model.png"), show=False)

    print("\nIniciando analise detalhada de erros por faixa de preco...")
    analyze_regression_error_by_segment(y_final_log, y_pred_log, price_original=train_prep['SalePrice'])

    # 7. Conjunto de Teste e Submissao Kaggle
    print_separator("Fase 6: Geracao de Submissao Final")
    
    print(f"Carregando dados de teste...")
    test_df = read_data(TEST_PATH)
    if not test_df.empty:
        test_cleaned = clean_and_impute_data(test_df)
        test_nb, _ = group_neighborhoods_by_price(test_cleaned, mapping=nb_mapping)
        test_feat = add_feature_engineering(test_nb)
        test_prep = preparar_dados_para_xgboost(test_feat)
        
        submission = gerar_submissao(modelo, test_prep, train_columns=train_columns, output_path=SUBMISSION_PATH)
        print(f"\nRESUMO DA SUBMISSAO (Primeiras 10 linhas):")
        print(submission.head(10).to_string(index=False))
    else:
        print("Dados de teste nao encontrados.")

    end_time = datetime.now()
    duration = end_time - start_time
    print_separator("Pipeline Finalizado")
    print(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duracao total: {duration}")
    print("-"*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERRO DURANTE A EXECUCAO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
