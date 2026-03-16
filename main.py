import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Importações dos módulos internos do projeto
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
    plot_feature_relationships
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

def print_separator(title):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80)

def main():
    # 1. Configurações de Caminhos e Ambiente
    TRAIN_PATH = "data/raw/train.csv"
    TEST_PATH = "data/raw/test.csv"
    PROCESSED_DIR = "data/processed"
    FIGURES_DIR = "reports/figures"
    DIST_DIR = os.path.join(FIGURES_DIR, "distributions")
    SUBMISSION_PATH = os.path.join(PROCESSED_DIR, "submission.csv")
    
    # Garantir que os diretórios existam
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DIST_DIR, exist_ok=True)
    
    start_time = datetime.now()
    print_separator("Iniciando Pipeline de House Price Discovery")
    print(f"🕒 Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. Carregamento de Dados
    # ---------------------------------------------------------
    print_separator("Fase 1: Carregamento de Dados")
    train_df = read_data(TRAIN_PATH)
    if train_df.empty:
        print("❌ Erro crítico: Dados de treino não encontrados.")
        return

    # Inserir análise de outliers aqui
    train_df = handle_outliers(train_df, is_train=True)

    print(f"\nℹ️  ESTATÍSTICAS DO ALVO (SalePrice):")
    print(train_df['SalePrice'].describe().to_string())

    # 3. Limpeza, Imputação e Feature Engineering
    # ---------------------------------------------------------
    print_separator("Fase 2: Preparação de Dados")
    
    # Relatório de Justificativa (Dica do Log)
    print("\n📝 ESTRATÉGIA DE LIMPEZA E IMPUTAÇÃO:")
    print(" - Categóricos (Alley, BsmtQual, etc): 'NA' significa ausência da característica.")
    print(" - GarageYrBlt: Preenchido com 0 se sem garagem, ou YearBuilt se possui (garagem costuma ser original).")
    print(" - LotFrontage: Mediana do Bairro (vizinhança define o padrão de recuo do lote).")
    print(" - Fallback: Mediana para numéricos e Moda para categóricos (preserva tendência central).")

    train_cleaned = clean_and_impute_data(train_df)
    
    # Agrupar bairros por preço antes da feature engineering (usando 5 níveis para comparação)
    train_nb, nb_mapping = group_neighborhoods_by_price(train_cleaned, bins=5)
    
    train_feat = add_feature_engineering(train_nb)
    
    # 4. Análise Exploratória (EDA) e Correlações
    # ---------------------------------------------------------
    print_separator("Fase 3: Análise Exploratória e Correlações")
    
    # Importância via Informação Mútua (MI)
    print("\n📊 CALCULANDO INFORMAÇÃO MÚTUA (MI)...")
    mi_scores = get_mutual_info_scores(train_feat)
    print(f"✔️ TOP 30 ATRIBUTOS POR MI (Útil para análise de agentes):")
    print(mi_scores.head(30).to_string())
    
    # Salvar Gráficos de Análise
    print("\n📉 Gerando e salvando visualizações...")
    plot_informacao_mutua(mi_scores, top_n=30, save_path=os.path.join(FIGURES_DIR, "informacao_mutua.png"), show=False)
    plot_correlation_heatmap(train_feat, target_column='SalePrice', top_n=15, save_path=os.path.join(FIGURES_DIR, "heatmap_correlacao.png"), show=False)
    
    # Gráficos de Relação Direta (Dica do Log)
    top_3_num = mi_scores.drop(['SalePrice'], errors='ignore').head(3).index.tolist()
    plot_feature_relationships(train_feat, 'SalePrice', top_3_num, save_dir=FIGURES_DIR, show=False)
    
    # Plots de distribuição para as Top 10 features
    top_10_features = mi_scores.head(10).index.tolist()
    num_cols = train_feat[top_10_features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = train_feat[top_10_features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"🔍 Gerando plots de distribuição em {DIST_DIR}...")
    plot_distributions(train_feat, numerical_cols=num_cols, categorical_cols=cat_cols, save_dir=DIST_DIR, show=False)

    # 5. Mineração de Dados (Fase 2 do Projeto)
    # ---------------------------------------------------------
    print_separator("Fase 4: Mineração de Padrões Frequentes")
    
    # Adicionando Clusterização de Mercado (Segmentação)
    train_clust = perform_clustering(train_feat, n_clusters=5)
    
    print("\n💠 Discretizando dados numéricos para mineração (5 Níveis)...")
    df_disc = discretize_numeric(train_clust, bins=5)
    
    # Regras de Associação
    print("📜 Minerando Regras de Associação (Apriori)...")
    # Incluímos MarketSegment e NbPriceTier nas colunas relevantes
    relevant_cols_mining = mi_scores.head(20).index.tolist() + ['SalePriceBin', 'MarketSegment', 'NbPriceTier']
    transactions = dataframe_to_transactions(df_disc[relevant_cols_mining])
    
    # Baixamos um pouco o suporte (0.05) para pegar regras mais específicas e subimos a confiança
    freq, rules, rules_target = mine_association_rules(transactions, min_sup=0.08, min_conf=0.75)
    
    if not rules_target.empty:
        print(f"✔️ {len(rules_target)} Regras encontradas com o alvo (SalePriceBin).")
        print("\n🔝 TOP 20 REGRAS DE ASSOCIAÇÃO (Ordenadas por Lift):")
        print(rules_target[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
              .sort_values('lift', ascending=False).head(20).to_string(index=False))
    else:
        print("⚠️ Nenhuma regra de associação forte encontrada com os parâmetros atuais.")

    # Padrões Sequenciais
    print("\n🧵 Minerando Padrões Sequenciais (PrefixSpan)...")
    # 1. Sequência por atributos da própria casa
    top_10_mi_names = sorted(mi_scores.head(10).index.tolist() + ['MarketSegment', 'NbPriceTier'])
    sequences = df_disc[top_10_mi_names].astype(str).values.tolist()
    seq_patterns = mine_sequential_patterns(sequences, min_sup_ratio=0.1)
    
    print(f"✔️ {len(seq_patterns)} Padrões de atributos encontrados.")
    print("🔝 TOP 10 PADRÕES DE ATRIBUTOS:")
    for supp, pat in seq_patterns[:10]:
        print(f"   Suporte: {supp:>4} | Padrão: {pat}")

    # 2. Evolução Temporal por Bairro (Dica do Log)
    # Preservamos o YearBuilt numérico para o cálculo da década
    df_disc['YearBuilt_Num'] = train_feat['YearBuilt'] 
    evol_patterns = mine_neighborhood_evolution(df_disc, min_sup_ratio=0.2)
    print(f"✔️ {len(evol_patterns)} Padrões de evolução temporal encontrados.")
    print("🔝 TOP 10 PADRÕES DE EVOLUÇÃO (Bairros):")
    for supp, pat in evol_patterns[:10]:
        print(f"   Suporte: {supp:>4} | Evolução: {pat}")

    # 6. Treinamento e Avaliação do Modelo (Fase 3 do Projeto)
    # ---------------------------------------------------------
    print_separator("Fase 5: Modelagem Preditiva (XGBoost)")
    
    print("\n⚙️  Preparando dados para o modelo...")
    train_prep = preparar_dados_para_xgboost(train_feat)
    
    print("\n🏋️  Iniciando treinamento com Validação Cruzada (10-fold)...")
    modelo, train_columns = treinar_e_avaliar_xgboost(train_prep)
    
    # Plotagem dos resultados finais de regressão
    print("\n📉 Gerando gráfico de resultados da regressão...")
    X_final = train_prep.drop(columns=['Id', 'SalePrice'], errors='ignore')
    y_final_log = np.log1p(train_prep['SalePrice'])
    y_pred_log = modelo.predict(X_final)
    
    plot_regression_results(y_final_log, y_pred_log, title="XGBoost Final (Treino/Log)", 
                           save_path=os.path.join(FIGURES_DIR, "resultados_regressao.png"), show=False)

    # Gráfico de Importância das Features (usando o do modelo)
    importances = modelo.feature_importances_
    feat_imp_df = pd.DataFrame({'Importance': importances}, index=X_final.columns).sort_values(by='Importance', ascending=False)
    plot_feature_importance(feat_imp_df, top_n=20, save_path=os.path.join(FIGURES_DIR, "importancia_features_model.png"), show=False)

    # 7. Conjunto de Teste e Submissão Kaggle
    # ---------------------------------------------------------
    print_separator("Fase 6: Geração de Submissão Final")
    
    print(f"📂 Carregando dados de teste...")
    test_df = read_data(TEST_PATH)
    if not test_df.empty:
        test_cleaned = clean_and_impute_data(test_df)
        
        # Aplicar o mapeamento de bairros criado no treino
        test_nb, _ = group_neighborhoods_by_price(test_cleaned, mapping=nb_mapping)
        
        test_feat = add_feature_engineering(test_nb)
        test_prep = preparar_dados_para_xgboost(test_feat)
        
        submission = gerar_submissao(modelo, test_prep, train_columns=train_columns, output_path=SUBMISSION_PATH)
        print(f"\n📊 RESUMO DA SUBMISSÃO (Primeiras 10 linhas):")
        print(submission.head(10).to_string(index=False))
    else:
        print("⚠️ Dados de teste não encontrados. Ignorando fase de submissão.")

    end_time = datetime.now()
    duration = end_time - start_time
    print_separator("Pipeline Finalizado")
    print(f"🕒 Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duração total: {duration}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERRO DURANTE A EXECUÇÃO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
