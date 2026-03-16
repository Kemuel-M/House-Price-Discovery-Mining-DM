import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

def plot_distributions(df: pd.DataFrame, numerical_cols: list, categorical_cols: list, save_dir: str = None, show: bool = True):
    """Gera plots de histogramas, boxplots e countplots e exibe estatísticas no terminal."""
    sb.set_style("whitegrid")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("\n📊 ANÁLISE ESTATÍSTICA DE DISTRIBUIÇÃO:")
    for col in numerical_cols:
        skew = df[col].skew()
        kurt = df[col].kurt()
        print(f" - {col:15} | Média: {df[col].mean():10.2f} | Skew: {skew:6.2f} | Kurt: {kurt:6.2f}")
        
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
        if show:
            plt.show()
        else:
            plt.close()

def plot_informacao_mutua(mi_scores: pd.Series, top_n: int = 30, save_path: str = None, show: bool = True):
    """Plota a Informação Mútua e já assume que o print foi feito no main."""
    plt.figure(figsize=(10, 12))
    top_mi = mi_scores.head(top_n)
    sb.set_style("whitegrid")
    sb.barplot(x=top_mi.values, y=top_mi.index, hue=top_mi.index, palette="viridis", legend=False)
    plt.title(f'Top {top_n} Atributos Mais Relevantes (Informação Mútua)', fontsize=16)
    plt.xlabel('Pontuação de Informação Mútua', fontsize=12)
    plt.ylabel('Atributos', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📉 Gráfico de MI salvo em: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, target_column: str, top_n: int = 15, save_path: str = None, show: bool = True):
    """Gera um heatmap e lista as maiores correlações no terminal."""
    numeric_df = df.select_dtypes(include=np.number)
    numeric_corr = numeric_df.corr()[target_column].sort_values(ascending=False)
    
    print(f"\n🔗 TOP {top_n} CORRELAÇÕES LINEARES COM {target_column}:")
    print(numeric_corr.head(top_n).to_string())

    top_corr_features = numeric_corr.abs().sort_values(ascending=False).head(top_n).index
    top_corr_matrix = numeric_df[top_corr_features].corr()
    
    plt.figure(figsize=(12, 10))
    sb.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title(f'Matriz de Correlação - Top {top_n} Features com {target_column}')
    
    if save_path:
        plt.savefig(save_path)
        print(f"📉 Heatmap de correlação salvo em: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, save_path: str = None, show: bool = True):
    """Plota a importância das features."""
    plt.figure(figsize=(10, 8))
    if isinstance(importance_df, pd.DataFrame):
        sb.barplot(x=importance_df.iloc[:top_n, 0], y=importance_df.index[:top_n])
    else:
        sb.barplot(x=importance_df[:top_n], y=importance_df.index[:top_n])
        
    plt.title(f'Top {top_n} Importância das Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📉 Gráfico de importância salvo em: {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_feature_relationships(df: pd.DataFrame, target_column: str, features: list, save_dir: str = None, show: bool = True):
    """Gera gráficos de dispersão e calcula correlação de Pearson no terminal."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📈 ANÁLISE DE RELAÇÃO DIRETA (vs {target_column}):")
    for col in features:
        if col == target_column: continue
        
        # Calcular correlação se for numérica
        if df[col].dtype.kind in 'biufc':
            corr_val = df[col].corr(df[target_column])
            print(f" - {col:15} | Correlação de Pearson: {corr_val:6.3f}")
        
        plt.figure(figsize=(10, 6))
        if df[col].dtype.kind in 'biufc':
            sb.regplot(data=df, x=col, y=target_column, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        else:
            sb.boxplot(data=df, x=col, y=target_column)
            plt.xticks(rotation=45)
            
        plt.title(f'Relação: {col} vs {target_column}')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"relacao_{col}_vs_target.png"))
        if show:
            plt.show()
        else:
            plt.close()

def plot_regression_results(y_true, y_pred, title="Resultados da Regressão", save_path: str = None, show: bool = True):
    """Plota dispersão Real vs Previsto e distribuição de resíduos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real vs Previsto
    ax1.scatter(y_true, y_pred, alpha=0.5, color='crimson')
    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    ax1.plot([p1, p2], [p1, p2], 'b--')
    ax1.set_xlabel('Valores Reais')
    ax1.set_ylabel('Previsões')
    ax1.set_title(f'{title}: Real vs. Previsto')
    
    # Resíduos
    residuos = y_true - y_pred
    sb.histplot(residuos, kde=True, ax=ax2, color='purple')
    ax2.set_title(f'{title}: Distribuição de Resíduos')
    ax2.set_xlabel('Erro')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📉 Gráfico de resultados da regressão salvo em: {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()
