import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from prefixspan import PrefixSpan
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def discretize_numeric(df: pd.DataFrame, target: str = "SalePrice", bins: int = 5) -> pd.DataFrame:
    """Discretiza atributos numericos."""
    df = df.copy()
    
    if bins == 5:
        labels = ["Muito Baixo", "Baixo", "Medio", "Alto", "Muito Alto"]
    elif bins == 7:
        labels = ["Mto Baixo", "Baixo", "Medio-Baixo", "Medio", "Medio-Alto", "Alto", "Mto Alto"]
    else:
        labels = [f"Nivel {i}" for i in range(1, bins + 1)]

    if target in df.columns:
        df[f"{target}Bin"] = pd.qcut(df[target], q=bins, labels=labels, duplicates="drop")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [c for c in numeric_cols if c != target and c != 'Id']

    for col in numeric_cols:
        try:
            df[col] = pd.qcut(df[col], q=bins, labels=labels, duplicates="drop")
        except ValueError:
            df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            
    return df

def dataframe_to_transactions(df: pd.DataFrame) -> list:
    """Converte DataFrame para transacoes."""
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str).map(lambda v: f"{col}={v}")
    return df_str.values.tolist()

def perform_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Agrupa casas em clusters."""
    df_clust = df.copy()
    print(f"\n--- EXECUTANDO CLUSTERIZACAO (K-Means: {n_clusters} clusters) ---")
    
    features_to_cluster = [f for f in df_clust.select_dtypes(include=[np.number]).columns if f not in ['Id', 'SalePrice']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust[features_to_cluster].fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clust['MarketSegment'] = kmeans.fit_predict(X_scaled)
    df_clust['MarketSegment'] = df_clust['MarketSegment'].map(lambda x: f"Segment_{x}")
    
    print(f"Clusterizacao concluida. Segmentos: {df_clust['MarketSegment'].unique().tolist()}")
    return df_clust

def mine_association_rules(transactions: list, min_sup: float = 0.05, min_conf: float = 0.6):
    """Gera regras de associacao."""
    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_arr, columns=te.columns_)

    freq = apriori(trans_df, min_support=min_sup, use_colnames=True, max_len=5)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

    rules_target = rules[
        rules["consequents"].apply(lambda x: any(item.startswith("SalePriceBin=") for item in x))
    ]
    return freq, rules, rules_target

def mine_neighborhood_evolution(df: pd.DataFrame, min_sup_ratio: float = 0.3):
    """Minera evolucao temporal por bairro."""
    print("\n--- MINERANDO EVOLUCAO TEMPORAL POR BAIRRO (POR DECADA) ---")
    
    df_evol = df.copy()
    if 'YearBuilt_Num' in df_evol.columns:
        df_evol['DecadeBuilt'] = (df_evol['YearBuilt_Num'] // 10) * 10
    else:
        df_evol['DecadeBuilt'] = (pd.to_numeric(df_evol['YearBuilt'], errors='coerce') // 10) * 10
    
    df_evol = df_evol.dropna(subset=['DecadeBuilt', 'SalePriceBin'])
    sequences = []
    
    for nb in df_evol['Neighborhood'].unique():
        nb_data = df_evol[df_evol['Neighborhood'] == nb]
        if nb_data.empty: continue
        
        decade_evolution = nb_data.groupby('DecadeBuilt').agg({
            'OverallQual': lambda x: str(x.mode().iloc[0]) if not x.mode().empty else str(x.iloc[0]),
            'SalePriceBin': lambda x: str(x.mode().iloc[0]) if not x.mode().empty else str(x.iloc[0])
        }).sort_index()
        
        nb_seq = [f"Q:{row['OverallQual']}_P:{row['SalePriceBin']}" for _, row in decade_evolution.iterrows()]
        if len(nb_seq) >= 2:
            sequences.append(nb_seq)
    
    print(f"Geradas {len(sequences)} sequencias de evolucao.")
    if not sequences:
        return []

    ps = PrefixSpan(sequences)
    patterns = ps.frequent(max(1, int(min_sup_ratio * len(sequences))))
    return sorted([p for p in patterns if len(p[1]) >= 2], key=lambda x: (-len(x[1]), -x[0]))

def mine_sequential_patterns(sequences: list, min_sup_ratio: float = 0.05):
    """Minera padroes sequenciais."""
    ps = PrefixSpan(sequences)
    patterns = ps.frequent(max(1, int(min_sup_ratio * len(sequences))))
    return sorted(patterns, key=lambda x: (-len(x[1]), -x[0]))
