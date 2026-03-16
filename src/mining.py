import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from prefixspan import PrefixSpan

def discretize_numeric(df: pd.DataFrame, target: str = "SalePrice", bins: int = 5) -> pd.DataFrame:
    """Discretiza atributos numéricos em faixas categóricas (Quintis ou Decis)."""
    df = df.copy()
    
    # Lista estendida de labels para suportar até 10 níveis
    all_labels = [
        "Nível 1", "Nível 2", "Nível 3", "Nível 4", "Nível 5", 
        "Nível 6", "Nível 7", "Nível 8", "Nível 9", "Nível 10"
    ]
    
    # Se for 5 ou 7, usamos nomes mais amigáveis
    if bins == 5:
        labels = ["Muito Baixo", "Baixo", "Médio", "Alto", "Muito Alto"]
    elif bins == 7:
        labels = ["Mto Baixo", "Baixo", "Médio-Baixo", "Médio", "Médio-Alto", "Alto", "Mto Alto"]
    else:
        labels = all_labels[:bins]

    # Discretiza o alvo se presente
    if target in df.columns:
        df[f"{target}Bin"] = pd.qcut(df[target], q=bins, labels=labels, duplicates="drop")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [c for c in numeric_cols if c != target and c != 'Id']

    for col in numeric_cols:
        try:
            # Tenta dividir em porções iguais de registros (quantis)
            df[col] = pd.qcut(df[col], q=bins, labels=labels, duplicates="drop")
        except ValueError:
            # Fallback para divisões de largura igual se houver muitos valores repetidos
            # Recalcula os labels pois o qcut pode ter reduzido o número de bins efetivos
            df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            
    return df

def dataframe_to_transactions(df: pd.DataFrame) -> list:
    """Converte DataFrame para formato de transações (list de lists)."""
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str).map(lambda v: f"{col}={v}")
    return df_str.values.tolist()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Agrupa as casas em clusters baseados nas principais features numéricas."""
    df_clust = df.copy()
    print(f"\n=== EXECUTANDO CLUSTERIZAÇÃO (K-Means: {n_clusters} clusters) ===")
    
    # Selecionar as top features numéricas para clusterizar (evitar categóricas aqui)
    features_to_cluster = df_clust.select_dtypes(include=[np.number]).columns.tolist()
    # Remover IDs e Alvos
    features_to_cluster = [f for f in features_to_cluster if f not in ['Id', 'SalePrice']]
    
    # Escalonar os dados (K-Means é sensível à escala)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust[features_to_cluster].fillna(0))
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clust['MarketSegment'] = kmeans.fit_predict(X_scaled)
    df_clust['MarketSegment'] = df_clust['MarketSegment'].map(lambda x: f"Segment_{x}")
    
    print(f"✔️ Clusterização concluída. Segmentos criados: {df_clust['MarketSegment'].unique().tolist()}")
    return df_clust

def mine_association_rules(transactions: list, min_sup: float = 0.05, min_conf: float = 0.6):
    """Gera itens frequentes e regras de associação."""
    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_arr, columns=te.columns_)

    freq = apriori(trans_df, min_support=min_sup, use_colnames=True, max_len=5)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

    # Filtro para regras que levam ao SalePriceBin
    rules_target = rules[
        rules["consequents"].apply(lambda x: any(item.startswith("SalePriceBin=") for item in x))
    ]
    return freq, rules, rules_target

def mine_neighborhood_evolution(df: pd.DataFrame, min_sup_ratio: float = 0.3):
    """
    Minera a evolução das casas dentro dos bairros agrupadas por década.
    """
    print("\n=== MINERANDO EVOLUÇÃO TEMPORAL POR BAIRRO (AGRUPADO POR DÉCADA) ===")
    
    df_evol = df.copy()
    
    # Usar YearBuilt_Num para calcular a década (garantido pelo main.py)
    if 'YearBuilt_Num' in df_evol.columns:
        df_evol['DecadeBuilt'] = (df_evol['YearBuilt_Num'] // 10) * 10
    else:
        df_evol['DecadeBuilt'] = (pd.to_numeric(df_evol['YearBuilt'], errors='coerce') // 10) * 10
    
    df_evol = df_evol.dropna(subset=['DecadeBuilt', 'SalePriceBin'])
    
    neighborhoods = df_evol['Neighborhood'].unique()
    sequences = []
    
    for nb in neighborhoods:
        nb_data = df_evol[df_evol['Neighborhood'] == nb]
        if nb_data.empty: continue
        
        # Agrupar por década e pegar a moda
        decade_evolution = nb_data.groupby('DecadeBuilt').agg({
            'OverallQual': lambda x: str(x.mode().iloc[0]) if not x.mode().empty else str(x.iloc[0]),
            'SalePriceBin': lambda x: str(x.mode().iloc[0]) if not x.mode().empty else str(x.iloc[0])
        }).sort_index()
        
        nb_seq = []
        for _, row in decade_evolution.iterrows():
            item = f"Q:{row['OverallQual']}_P:{row['SalePriceBin']}"
            nb_seq.append(item)
        
        if len(nb_seq) >= 2:
            sequences.append(nb_seq)
    
    print(f"✔️ Geradas {len(sequences)} sequências de evolução por década.")
    
    if not sequences:
        print("⚠️ Nenhuma sequência de evolução encontrada.")
        return []

    ps = PrefixSpan(sequences)
    min_sup_abs = max(1, int(min_sup_ratio * len(sequences)))
    patterns = ps.frequent(min_sup_abs)
    
    long_patterns = [p for p in patterns if len(p[1]) >= 2]
    return sorted(long_patterns, key=lambda x: (-len(x[1]), -x[0]))

def mine_sequential_patterns(sequences: list, min_sup_ratio: float = 0.05):
    """Minera padrões sequenciais usando PrefixSpan."""
    ps = PrefixSpan(sequences)
    min_sup_abs = max(1, int(min_sup_ratio * len(sequences)))
    patterns = ps.frequent(min_sup_abs)
    return sorted(patterns, key=lambda x: (-len(x[1]), -x[0]))
