import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

def read_data(file_path: str) -> pd.DataFrame:
    """Carrega o CSV e retorna um DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dados carregados de {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return pd.DataFrame()

def handle_outliers(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Identifica e remove outliers usando Isolation Forest e regras de dominio.
    Nota: Só removemos outliers do conjunto de TREINO.
    """
    if not is_train:
        return df

    df_clean = df.copy()
    print("\n--- ANALISE DE OUTLIERS (TREINO) ---")
    
    # 1. Regras de Dominio (Recomendacao do autor)
    outliers_manual = df_clean[(df_clean['GrLivArea'] > 4000) | (df_clean['LotArea'] > 100000)].index
    if len(outliers_manual) > 0:
        print(f"Detectados {len(outliers_manual)} outliers via regras de dominio (GrLivArea/LotArea).")

    # 2. Isolation Forest (Deteccao estatistica multivariada)
    # Usamos as colunas numericas mais importantes para a deteccao
    cols_to_check = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea', 'LotArea', 'SalePrice']
    cols_present = [c for c in cols_to_check if c in df_clean.columns]
    
    iso = IsolationForest(contamination=0.01, random_state=42)
    preds = iso.fit_predict(df_clean[cols_present].fillna(df_clean[cols_present].median()))
    outliers_iso = df_clean.index[preds == -1]
    print(f"Detectados {len(outliers_iso)} outliers via Isolation Forest.")

    # Remover os indices identificados (uniao das tecnicas)
    to_remove = list(set(outliers_manual) | set(outliers_iso))
    df_clean = df_clean.drop(to_remove, axis=0)
    
    print(f"Remocao concluida. {len(to_remove)} registros removidos. Linhas restantes: {len(df_clean)}")
    
    return df_clean

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza e imputacao robusta de valores ausentes."""
    df_copy = df.copy()
    print("\n--- LIMPEZA E IMPUTACAO DE DADOS ---")

    # 1. Categoricos onde NaN significa 'Nao Possui'
    cols_fill_na = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
    ]
    for col in cols_fill_na:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('NA')

    # 2. Logica para GarageYrBlt
    if 'GarageType' in df_copy.columns and 'GarageYrBlt' in df_copy.columns:
        mask_no_garage = df_copy["GarageType"] == "NA"
        df_copy.loc[mask_no_garage, "GarageYrBlt"] = df_copy.loc[mask_no_garage, "GarageYrBlt"].fillna(0)
        df_copy.loc[~mask_no_garage, "GarageYrBlt"] = df_copy.loc[~mask_no_garage, "GarageYrBlt"].fillna(
            df_copy.loc[~mask_no_garage, "YearBuilt"]
        )

    # 3. Logica para LotFrontage (mediana do bairro)
    if 'LotFrontage' in df_copy.columns and 'Neighborhood' in df_copy.columns:
        df_copy['LotFrontage'] = df_copy.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # 4. Outros preenchimentos padrao
    if 'MasVnrArea' in df_copy.columns:
        df_copy['MasVnrArea'] = df_copy['MasVnrArea'].fillna(0)

    # Preencher com Moda para colunas com poucos faltantes
    low_missing_cols = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Functional']
    for col in low_missing_cols:
        if col in df_copy.columns and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    # FALLBACK GLOBAL: Garante que NENHUM nulo passe para o modelo
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            if df_copy[col].dtype.kind in 'biufc': # Numericos
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else: # Categoricos/Objetos
                mode_val = df_copy[col].mode()
                if not mode_val.empty:
                    df_copy[col] = df_copy[col].fillna(mode_val[0])
                else:
                    df_copy[col] = df_copy[col].fillna("NA")

    remaining_missing = df_copy.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Atencao: Ainda restam {remaining_missing} valores ausentes!")
    else:
        print(f"Limpeza concluida. Valores ausentes tratados.")
    
    return df_copy

def group_neighborhoods_by_price(df: pd.DataFrame, bins: int = 5, mapping: dict = None) -> (pd.DataFrame, dict):
    """Agrupa bairros em niveis de preco."""
    df_nb = df.copy()
    
    if mapping is None:
        print(f"\n--- AGRUPANDO BAIRROS POR PRECO MEDIO ({bins} Niveis) ---")
        nb_means = df_nb.groupby('Neighborhood')['SalePrice'].mean().sort_values()
        labels = [f"NB_Level_{i}" for i in range(1, bins + 1)]
        nb_groups = pd.qcut(nb_means, q=bins, labels=labels)
        mapping = nb_groups.to_dict()
        print("Mapeamento de Bairros criado.")
    
    df_nb['NbPriceTier'] = df_nb['Neighborhood'].map(mapping)
    if df_nb['NbPriceTier'].isnull().any():
        median_tier = f"NB_Level_{bins // 2 + 1}"
        df_nb['NbPriceTier'] = df_nb['NbPriceTier'].fillna(median_tier)

    return df_nb, mapping

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features baseadas no conhecimento de dominio."""
    df_feat = df.copy()
    print("\n--- EXECUTANDO FEATURE ENGINEERING ---")

    df_feat['TotalSF'] = df_feat['GrLivArea'] + df_feat['TotalBsmtSF']
    df_feat['QualAreaInteract'] = df_feat['OverallQual'] * df_feat['GrLivArea']
    df_feat['TotalBathrooms'] = (
        df_feat['FullBath'] + (0.5 * df_feat['HalfBath']) + 
        df_feat['BsmtFullBath'] + (0.5 * df_feat['BsmtHalfBath'])
    )

    ref_year = df_feat['YrSold'] if 'YrSold' in df_feat.columns else 2010
    df_feat['HouseAge'] = ref_year - df_feat['YearBuilt']
    df_feat['YearsSinceRemod'] = ref_year - df_feat['YearRemodAdd']
    df_feat['IsNewHouse'] = (df_feat['YearBuilt'] == df_feat['YrSold']).astype(int)

    df_feat['TotalPorchSF'] = (
        df_feat['WoodDeckSF'] + df_feat['OpenPorchSF'] + 
        df_feat['EnclosedPorch'] + df_feat['3SsnPorch'] + df_feat['ScreenPorch']
    )

    df_feat['HasPool'] = (df_feat['PoolArea'] > 0).astype(int)
    df_feat['Has2ndFloor'] = (df_feat['2ndFlrSF'] > 0).astype(int)
    df_feat['HasGarage'] = (df_feat['GarageArea'] > 0).astype(int)
    df_feat['HasFireplace'] = (df_feat['Fireplaces'] > 0).astype(int)

    print(f"Novas features criadas. Total de colunas: {len(df_feat.columns)}")
    return df_feat

def get_mutual_info_scores(df: pd.DataFrame, target_column: str = 'SalePrice') -> pd.Series:
    """Calcula MI scores para todas as colunas."""
    df_mi = df.copy()
    if 'Id' in df_mi.columns:
        df_mi = df_mi.drop('Id', axis=1)

    # Ajuste para compatibilidade futura (Pandas 3/4)
    cat_cols = df_mi.select_dtypes(include=['object', 'category', 'string']).columns
    for col in cat_cols:
        df_mi[col], _ = pd.factorize(df_mi[col])
    
    for col in df_mi.columns:
        if df_mi[col].isnull().any():
            df_mi[col] = df_mi[col].fillna(df_mi[col].median())

    X = df_mi.drop(columns=target_column)
    y = df_mi[target_column]
    
    mi_scores = mutual_info_regression(X, y, random_state=42)
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

def preparar_dados_para_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeamento ordinal e preparação para o suporte nativo do XGBoost."""
    df_proc = df.copy()
    
    ordinais_mapeamento = {
        'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
        'Utilities': {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0},
        'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0},
        'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0},
        'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0},
        'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0},
        'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'Functional': {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0},
        'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0},
        'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
        'PavedDrive': {'Y': 2, 'P': 1, 'N': 0},
        'PoolQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0},
        'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
    }

    for col, mapping in ordinais_mapeamento.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(mapping).fillna(0)

    # Converter todas as colunas de texto restantes para o tipo category
    # Ajuste para compatibilidade futura (Pandas 3/4)
    txt_cols = df_proc.select_dtypes(include=['object', 'string']).columns
    for col in txt_cols:
        df_proc[col] = df_proc[col].astype('category')
    
    # MSSubClass é numérico mas é categórico por natureza
    if 'MSSubClass' in df_proc.columns:
        df_proc['MSSubClass'] = df_proc['MSSubClass'].astype('category')

    return df_proc
