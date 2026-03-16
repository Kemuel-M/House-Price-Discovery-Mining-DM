import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

def plot_distributions(df: pd.DataFrame, numerical_cols: list, categorical_cols: list, save_dir: str = None, show: bool = True):
    """Gera plots de distribuicao e exibe estatisticas/contagens detalhadas no terminal."""
    sb.set_style("whitegrid")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("\n--- ANALISE DE DISTRIBUICAO: VARIAVEIS NUMERICAS ---")
    for col in numerical_cols:
        # Estatisticas
        stats = df[col].describe()
        skew = df[col].skew()
        print(f"\nAtributo: {col}")
        print(f"Estatisticas:\n{stats.to_string()}")
        print(f"Skewness: {skew:.2f} | Kurtosis: {df[col].kurt():.2f}")
        
        # Histograma em texto (Decis)
        decis = pd.cut(df[col], bins=10).value_counts().sort_index()
        print("Distribuicao por faixas (Bins):")
        print(decis.to_string())
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sb.histplot(df[col], kde=True, bins=30)
        plt.title(f'Histograma de {col}')
        plt.subplot(1, 2, 2)
        sb.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"dist_num_{col}.png"))
        if show: plt.show()
        else: plt.close()

    print("\n--- ANALISE DE DISTRIBUICAO: VARIAVEIS CATEGORICAS ---")
    for col in categorical_cols:
        counts = df[col].value_counts()
        perc = df[col].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'Contagem': counts, 'Percentual (%)': perc})
        
        print(f"\nAtributo: {col}")
        print(dist_df.to_string())
        
        plt.figure(figsize=(10, 6))
        sb.countplot(data=df, x=col, order=counts.index, hue=col, palette="viridis", legend=False)
        plt.xticks(rotation=45)
        plt.title(f'Distribuicao Categorica: {col}')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"dist_cat_{col}.png"))
        if show: plt.show()
        else: plt.close()

def plot_informacao_mutua(mi_scores: pd.Series, top_n: int = 30, save_path: str = None, show: bool = True):
    """Plota MI e detalha os scores no terminal."""
    top_mi = mi_scores.head(top_n)
    print(f"\n--- DADOS DE INFORMACAO MUTUA (Top {top_n}) ---")
    print(top_mi.to_string())

    plt.figure(figsize=(10, 12))
    sb.set_style("whitegrid")
    sb.barplot(x=top_mi.values, y=top_mi.index, hue=top_mi.index, palette="viridis", legend=False)
    plt.title(f'Top {top_n} Atributos Mais Relevantes (MI)')
    plt.xlabel('Pontuacao de Informacao Mutua')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grafico de MI salvo em: {save_path}")
    if show: plt.show()
    else: plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, target_column: str, top_n: int = 15, save_path: str = None, show: bool = True):
    """Gera heatmap e imprime as matrizes de correlacao (Pearson e Spearman) no terminal."""
    numeric_df = df.select_dtypes(include=np.number)
    
    # Pearson
    pearson_corr = numeric_df.corr()[target_column].sort_values(ascending=False)
    # Spearman (Captura relacoes nao-lineares monotonicamente crescentes/decrescentes)
    spearman_corr = numeric_df.corr(method='spearman')[target_column].sort_values(ascending=False)
    
    top_corr_features = pearson_corr.abs().sort_values(ascending=False).head(top_n).index
    top_corr_matrix = numeric_df[top_corr_features].corr()
    
    print(f"\n--- MATRIZ DE CORRELACAO DETALHADA (Top {top_n} vs {target_column}) ---")
    corr_summary = pd.DataFrame({
        'Pearson': pearson_corr[top_corr_features],
        'Spearman': spearman_corr[top_corr_features]
    })
    print(corr_summary.to_string())

    plt.figure(figsize=(12, 10))
    sb.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f'Matriz de Correlacao (Pearson) - Top {top_n} Features')
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap salvo em: {save_path}")
    if show: plt.show()
    else: plt.close()

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, save_path: str = None, show: bool = True):
    """Plota importancia e imprime a tabela de importancia no terminal."""
    print(f"\n--- TABELA DE IMPORTANCIA DAS FEATURES (Top {top_n}) ---")
    print(importance_df.head(top_n).to_string())

    plt.figure(figsize=(10, 8))
    sb.barplot(x=importance_df.iloc[:top_n, 0], y=importance_df.index[:top_n], hue=importance_df.index[:top_n], palette="magma", legend=False)
    plt.title(f'Top {top_n} Importancia das Features (Modelo)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grafico de importancia salvo em: {save_path}")
    if show: plt.show()
    else: plt.close()

def plot_feature_relationships(df: pd.DataFrame, target_column: str, features: list, save_dir: str = None, show: bool = True):
    """Gera graficos de dispersao/boxplots e imprime resumos de agregacao e correlacoes (Pearson/Spearman)."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- ANALISE DE RELACAO DETALHADA (Target: {target_column}) ---")
    for col in features:
        if col == target_column: continue
        
        print(f"\nRelacao: {col} vs {target_column}")
        if df[col].dtype.kind in 'biufc':
            corr_p = df[col].corr(df[target_column], method='pearson')
            corr_s = df[col].corr(df[target_column], method='spearman')
            print(f"Correlacao -> Pearson: {corr_p:.4f} | Spearman: {corr_s:.4f}")
            
            # Resumo por quartis da feature
            quartis = pd.qcut(df[col], q=4, duplicates='drop')
            resumo = df.groupby(quartis, observed=True)[target_column].agg(['mean', 'median', 'std', 'count'])
            print("Estatisticas do Alvo por Quartis da Feature:")
            print(resumo.to_string())
        else:
            resumo = df.groupby(col, observed=True)[target_column].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)
            print("Estatisticas do Alvo por Categoria:")
            print(resumo.to_string())
        
        plt.figure(figsize=(10, 6))
        if df[col].dtype.kind in 'biufc':
            sb.regplot(data=df, x=col, y=target_column, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        else:
            sb.boxplot(data=df, x=col, y=target_column)
            plt.xticks(rotation=45)
        plt.title(f'Relacao: {col} vs {target_column}')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"relacao_{col}_vs_target.png"))
        if show: plt.show()
        else: plt.close()

def analyze_regression_error_by_segment(y_true, y_pred, price_original=None):
    """Analisa o erro do modelo segmentado por faixas de preco."""
    residuos_abs = np.abs(y_true - y_pred)
    df_err = pd.DataFrame({
        'Real': y_true,
        'Previsto': y_pred,
        'Erro_Abs_Log': residuos_abs
    })
    
    if price_original is not None:
        df_err['Preco_Original'] = price_original
        # Criar decis baseados no preco original
        df_err['Faixa_Preco'] = pd.qcut(df_err['Preco_Original'], q=10, labels=[f"D{i+1}" for i in range(10)])
    else:
        # Criar decis baseados no log do preco real
        df_err['Faixa_Preco'] = pd.qcut(df_err['Real'], q=10, labels=[f"D{i+1}" for i in range(10)])

    print("\n--- ANALISE DE ERRO POR DECIL DE PRECO ---")
    print("D1: 10% mais baratas -> D10: 10% mais caras")
    
    segmentacao = df_err.groupby('Faixa_Preco', observed=True).agg(
        Media_Erro_Log=('Erro_Abs_Log', 'mean'),
        Mediana_Erro_Log=('Erro_Abs_Log', 'median'),
        Max_Erro_Log=('Erro_Abs_Log', 'max'),
        Contagem=('Real', 'count')
    )
    
    if price_original is not None:
        # Calcular MAE na escala original por segmento
        df_err['Erro_Original'] = np.abs(np.expm1(y_true) - np.expm1(y_pred))
        mae_orig = df_err.groupby('Faixa_Preco', observed=True)['Erro_Original'].mean()
        segmentacao['MAE_Original ($)'] = mae_orig
    
    print(segmentacao.to_string())
    
    # Identificar o pior decil
    pior_decil = segmentacao['Media_Erro_Log'].idxmax()
    print(f"\nO modelo apresenta o MAIOR erro medio no decil: {pior_decil}")

def plot_regression_results(y_true, y_pred, title="Resultados da Regressao", save_path: str = None, show: bool = True):
    """Plota resultados e imprime metricas e decis de erro no terminal."""
    residuos = y_true - y_pred
    
    print(f"\n--- METRICAS DE REGRESSAO: {title} ---")
    mse = np.mean(residuos**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuos))
    print(f"RMSE (log): {rmse:.4f}")
    print(f"MAE (log):  {mae:.4f}")
    
    print("\nAnalise de Residuos (Decis de Erro):")
    res_series = pd.Series(residuos)
    print(res_series.describe().to_string())

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, color='crimson')
    p1, p2 = max(max(y_pred), max(y_true)), min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsoes')
    plt.title(f'Real vs. Previsto')
    
    plt.subplot(1, 3, 2)
    sb.histplot(residuos, kde=True, color='purple')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'Distribuicao de Residuos')
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_pred, residuos, alpha=0.5, color='teal')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Previsoes')
    plt.ylabel('Residuos')
    plt.title(f'Residuos vs. Previsoes')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grafico de regressao salvo em: {save_path}")
    if show: plt.show()
    else: plt.close()
