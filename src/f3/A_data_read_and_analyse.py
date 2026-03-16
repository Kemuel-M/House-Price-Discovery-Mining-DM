import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def read_data(file_path:str) -> pd.DataFrame:
    # Carregar os dados
    try:
        # Tente carregar do seu ambiente local
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Arquivo não encontrado. Certifique-se de que estão no diretório correto.")
        print(f"Diretorio do arquivo mandado {file_path}")
        print(f"Diretorio atual {os.getcwd()}")
    # Visualizar as primeiras linhas do conjunto de treino
    if not df.empty:
        print("Amostra do Conjunto")
        print(df.head())

    return df


def analyse_data(df: pd.DataFrame) -> None:
    """ Função para analisar os dados exploratoriamente, exibindo informações gerais e estatísticas do dataset."""
    print("\n=== ANÁLISE EXPLORATÓRIA DOS DADOS ===")

    # Exibir informações gerais do DataFrame
    print(f"Shape do dataset: {df.shape}")
    print(f"Colunas: \n{list(df.columns)}")
    print("\nInformações do DataFrame:")
    print(df.info())
    print("\nAnálise dos Dados:")
    print(df.describe())
    print("\nnValores Ausentes:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # Verificar a distribuição de valores únicos por coluna
    max_unique_values = 50
    print("\nDistribuição de Valores Únicos por Coluna:")
    unique_counts = df.nunique()
    print(f"Valores únicos por coluna, menor que {max_unique_values}:")
    print(unique_counts[unique_counts < max_unique_values])
    print(f"\nDistribuição de Valores Únicos por Coluna, menor que {max_unique_values}:")
    for coluna in df.columns:
        n_unique = df[coluna].nunique()
        if n_unique < max_unique_values:
            print(f"-> Coluna '{coluna}' ({n_unique} valores únicos):")
            print(df[coluna].unique().tolist())


'''
- Verificar distribuicao de variaveis numericas
- Verificar distribuicao de variaveis categoricas
- Verificar outliers
'''

def analyze_distributions(df: pd.DataFrame) -> None:
    """
    Realiza a análise de distribuição de variáveis numéricas e categóricas,
    imprimindo estatísticas no terminal e gerando plots.
    """
    # 1. Separar colunas numéricas de categóricas
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Remover colunas de ID ou com alta cardinalidade que não fazem sentido analisar individualmente
    if 'Id' in numerical_cols:
        numerical_cols.remove('Id')

    print("\n=== ANÁLISE DE DISTRIBUIÇÃO DAS VARIÁVEIS ===")

    # 2. Análise de Variáveis Numéricas (Terminal)
    print("\n--- Análise de Variáveis Numéricas (Terminal) ---")
    for col in numerical_cols:
        skewness = df[col].skew()
        kurt = df[col].kurt()
        print(f"-> Coluna '{col}':")
        print(f"   - Assimetria (Skewness): {skewness:.2f}")
        print(f"   - Curtose (Kurtosis): {kurt:.2f}")

    # 3. Análise de Variáveis Categóricas (Terminal)
    print("\n--- Análise de Variáveis Categóricas (Terminal) ---")
    for col in categorical_cols:
        print(f"\n-> Contagem de valores para a coluna '{col}':")
        # Mostra a contagem e a porcentagem
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        # Combina as duas séries para uma exibição clara
        result_df = pd.DataFrame({'Contagem': counts, 'Porcentagem (%)': percentages.round(2)})
        print(result_df)
        print("-" * 30)

    # --- PLOTS ---
    if False:
        # Define um estilo visual mais agradável para os gráficos
        sb.set_style("whitegrid")

        # 4. Plots para Variáveis Numéricas
        print("\nGerando plots para variáveis numéricas...")
        for col in numerical_cols:
            plt.figure(figsize=(12, 5))

            # Histograma com KDE
            plt.subplot(1, 2, 1)
            sb.histplot(df[col], kde=True, bins=30)
            plt.title(f'Histograma de {col}')

            # Boxplot
            plt.subplot(1, 2, 2)
            sb.boxplot(x=df[col])
            plt.title(f'Boxplot de {col}')
            
            plt.tight_layout() # Ajusta o layout para evitar sobreposição
            plt.show()

        # 5. Plots para Variáveis Categóricas
        print("\nGerando plots para variáveis categóricas...")
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            # Ordena as categorias pela frequência para melhor visualização
            order = df[col].value_counts().index
            sb.countplot(y=df[col], order=order)
            plt.title(f'Distribuição de {col}')
            plt.xlabel('Contagem')
            plt.ylabel(col)
            
            plt.tight_layout() # Ajusta o layout para evitar sobreposição
            plt.show()

'''
- Analisar correlacoes
- Analisar correlacoes entre variaveis (todas, categorias e numericas, informacao mutua) e target (SalePrice)
- Analisar a correlação tanto boa quanto ruim, em relacao ao SalePrice, e retornar uma lista das features em ordem de importancia
- fazer uma matriz de correlacao, e prints no terminal das correlacoes tambem.
'''

def analyse_correlations(df: pd.DataFrame, target_column: str = 'SalePrice') -> pd.DataFrame:
    """
    Função para analisar a correlação de todas as features com a coluna alvo,
    gerando visualizações e retornando um ranking de importância.

    Args:
        df: DataFrame contendo os dados.
        target_column: O nome da coluna alvo.

    Returns:
        Um DataFrame com as features e suas respectivas importâncias (informação mútua),
        ordenado da mais importante para a menos importante.
    """
    print(f"\n=== ANÁLISE DE CORRELAÇÃO COM '{target_column}' ===")
    df_corr = df.copy()
    max_correlation_features = 40

    # --- 1. Correlação de Pearson e Heatmap para Variáveis Numéricas ---
    print("\n1. Análise de Correlação de Pearson (Variáveis Numéricas):")
    numeric_df = df_corr.select_dtypes(include=np.number)
    
    # Imprime as correlações mais fortes e mais fracas no console
    numeric_corr = numeric_df.corr()[target_column].sort_values(ascending=False)
    print("\nCorrelações mais fortes (positivas e negativas):")
    print(numeric_corr.head(max_correlation_features//2))
    print("\nCorrelações mais fracas (próximas de zero):")
    print(numeric_corr.tail(max_correlation_features//2))

    # Gerar o Heatmap
    print("\nGerando Heatmap de Correlação para as features numéricas mais importantes...")
    # Selecionar as features com maior correlação (em valor absoluto) com o alvo
    top_corr_features = numeric_corr.abs().sort_values(ascending=False).head(max_correlation_features).index
    top_corr_matrix = numeric_df[top_corr_features].corr()
    
    plt.figure(figsize=(12, 10))
    sb.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title(f'Matriz de Correlação das 15 Features Mais Correlacionadas com {target_column}')
    plt.show()

    # --- 2. Informação Mútua para Todas as Variáveis (Numéricas e Categóricas) ---
    print("\n2. Análise de Informação Mútua (Todas as Variáveis):")
    
    df_mi = df_corr.drop(columns=['Id']) # 'Id' não é uma feature útil
    
    # Corrigindo o FutureWarning: preenchendo NAs de forma segura
    for col in df_mi.select_dtypes(include=['object', 'category']).columns:
        if df_mi[col].isnull().any():
            df_mi[col] = df_mi[col].cat.add_categories("missing").fillna("missing")
        df_mi[col] = LabelEncoder().fit_transform(df_mi[col])
    for col in df_mi.select_dtypes(include=np.number).columns:
        # Não preencher o alvo
        if col != target_column:
            df_mi[col] = df_mi[col].fillna(df_mi[col].median())

    # Codificar todas as colunas categóricas
    for col in df_mi.select_dtypes(include='object').columns:
        df_mi[col] = LabelEncoder().fit_transform(df_mi[col])
        
    X = df_mi.drop(columns=target_column)
    y = df_mi[target_column]

    mutual_info = mutual_info_regression(X, y, random_state=42)
    mutual_info_series = pd.Series(mutual_info, index=X.columns, name="MutualInformation")
    mutual_info_series = mutual_info_series.sort_values(ascending=False)
    
    print("\nTop Features por Informação Mútua:")
    print(mutual_info_series.head(max_correlation_features))

    # Gerar o gráfico de barras para a Informação Mútua
    print("\nGerando Gráfico de Barras para a Importância das Features (Informação Mútua)...")
    top_mi_features = mutual_info_series.head(max_correlation_features) # Visualizar as top
    plt.figure(figsize=(10, 8))
    sb.barplot(x=top_mi_features.values, y=top_mi_features.index)
    plt.title(f'Top Features Mais Importantes em Relação a {target_column} (Informação Mútua)')
    plt.xlabel('Pontuação de Informação Mútua')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    # --- 3. Consolidar e Retornar a Lista de Features por Importância ---
    print(f"\n3. Ranking Final de Features em Relação a '{target_column}':")
    feature_importance_df = pd.DataFrame(mutual_info_series).reset_index()
    feature_importance_df.columns = ['Feature', 'Importance (Mutual Information)']
    print(feature_importance_df.head(max_correlation_features))

    return feature_importance_df


