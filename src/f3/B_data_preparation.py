import pandas as pd
import numpy as np

'''
- Corrigir valores ausentes:
    se um valor ausente significa que nao tem, entao crie uma nova categoria 'NA' (categoricos) ou 0 (numerico) dizendo que nao tem.
Alguns exemplos de extrategias para lidar com valores ausentes:
    BsmtQual      : 'NA'  (descrição já prevê esse rótulo = *No Basement*)
    GarageType    : 'NA'  (descrição já prevê esse rótulo = *No Garage*)
    FireplaceQu   : 'NA'  (descrição já prevê esse rótulo = *No Fireplace*)
    GarageYrBlt   :  
        Se a casa possui garagem → usa `YearBuilt` (garagem geralmente construída junto)  
        Se não possui garagem → preenche 0 (código neutro)
    LotFrontage   : mediana do bairro (`Neighborhood`) — mantém coerência espacial.
- Aplicar outras tecnicas de imputação nos outros valores ausentes. Deve ser feito em TODAS as features que tem valores ausentes.
'''

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para limpar e imputar valores ausentes em um DataFrame de dados imobiliários.

    A função aplica as seguintes estratégias:
    - Preenche com 'NA' para características categóricas onde a ausência é significativa (ex: 'Alley', 'BsmtQual').
    - Preenche com 'None' ou 0 para características de alvenaria ('MasVnrType', 'MasVnrArea').
    - Preenche 'LotFrontage' com a mediana do bairro ('Neighborhood').
    - Preenche 'GarageYrBlt' com 0, pois a ausência indica que não há garagem.
    - Preenche o valor ausente em 'Electrical' com o valor mais frequente (moda).

    Args:
        df: DataFrame original com valores ausentes.

    Returns:
        DataFrame com os valores ausentes tratados.
    """
    df_copy = df.copy()

    print("\n=== LIMPEZA E IMPUTAÇÃO DE DADOS ===")
    # --- 1. Categóricos onde NaN significa 'Não Possui' ---
    # Lista completa de colunas para preencher com 'NA'
    cols_fill_na = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
    ]
    for col in cols_fill_na:
        df_copy[col] = df_copy[col].fillna('NA')

    # --- 2. Lógica Sofisticada para 'GarageYrBlt' ---
    # Primeiro, identificamos as casas que não têm garagem (após a etapa anterior)
    mask_no_garage = df_copy["GarageType"] == "NA"
    
    # Para casas SEM garagem, preenchemos NaN em 'GarageYrBlt' com 0
    df_copy.loc[mask_no_garage, "GarageYrBlt"] = df_copy.loc[mask_no_garage, "GarageYrBlt"].fillna(0)
    
    # Para casas COM garagem, preenchemos NaN em 'GarageYrBlt' com o ano de construção da casa
    df_copy.loc[~mask_no_garage, "GarageYrBlt"] = df_copy.loc[~mask_no_garage, "GarageYrBlt"].fillna(
        df_copy.loc[~mask_no_garage, "YearBuilt"]
    )

    # --- 3. Lógica Espacial para 'LotFrontage' ---
    # Usa a mediana do bairro para manter a coerência
    df_copy['LotFrontage'] = df_copy.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    # --- 4. Lógica para Alvenaria ---
    # Se o tipo é ausente, preenche com 'NA' e a área com 0
    df_copy['MasVnrType'] = df_copy['MasVnrType'].fillna('NA')
    df_copy['MasVnrArea'] = df_copy['MasVnrArea'].fillna(0)

    # --- 5. Imputação por Moda para 'Electrical' ---
    # Preenche o único valor ausente com o mais comum
    electrical_mode = df_copy['Electrical'].mode()[0]
    df_copy['Electrical'] = df_copy['Electrical'].fillna(electrical_mode)

    remaining_missing = df_copy.isnull().sum().sum()
    if remaining_missing == 0:
        print("Todos os valores ausentes foram tratados com sucesso!")
    else:
        print(f"Ainda restam {remaining_missing} valores ausentes.")

    return df_copy

'''
- O XGBoost, em sua essência, é um algoritmo que trabalha com números.[1] Portanto, nosso principal objetivo é converter todas as suas features em um formato numérico, fazendo isso de uma maneira que o modelo consiga extrair o máximo de informação.
Features Numéricas: Variáveis que já são números.
Features Categóricas Ordinais: Variáveis que representam categorias com uma ordem ou hierarquia clara.
Features Categóricas Nominais: Variáveis que representam categorias sem uma ordem intrínseca.

- Função para preparar os dados para o XGBoost:
Identifique os Tipos: Separe suas colunas nos três grupos: Numéricas, Categóricas Ordinais e Categóricas Nominais.
as Numéricas ja foi tratado na etapa de limpeza e imputação.
Trate as Ordinais: Use o mapeamento manual (com dicionários) para converter todas as features ordinais em números que preservem sua hierarquia.
Trate as Nominais: A melhor abordagem é usar o suporte nativo do XGBoost, É eficiente e simples.
    - Converta as colunas categóricas do seu DataFrame para o tipo category do Pandas. Ao instanciar o modelo XGBoost (XGBRegressor), passe o parâmetro enable_categorical=True.
    - Se preferir a abordagem mais tradicional, use o One-Hot Encoding.
'''

def preparar_dados_para_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara um DataFrame para treinamento com XGBoost, tratando colunas
    ordinais e nominais conforme especificado.

    Args:
        df: O DataFrame contendo os dados brutos.

    Returns:
        O DataFrame processado e pronto para o XGBoost.
    """
    # Copia o DataFrame para evitar modificações no original
    df_proc = df.copy()

    print("\n=== PRÉ-PROCESSAMENTO PARA XGBOOST ===")
    # 1. DEFINIÇÃO DAS COLUNAS
    
    # Colunas Categóricas Nominais (serão convertidas para o tipo 'category')
    nominais = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
        'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 
        'MiscFeature', 'SaleType', 'SaleCondition'
    ]

    # Colunas Categóricas Ordinais e seus respectivos mapeamentos
    ordinais_mapeamento = {
        'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
        'Utilities': {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0},
        'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0},
        'OverallQual': {i: i for i in range(1, 11)},
        'OverallCond': {i: i for i in range(1, 11)},
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

    # 2. TRATAMENTO DAS ORDINAIS
    print("Mapeando colunas ordinais...")
    for col, mapping in ordinais_mapeamento.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(mapping)
        else:
            print(f"Aviso: Coluna ordinal '{col}' não encontrada no DataFrame.")
            
    # 3. TRATAMENTO DAS NOMINAIS
    print("Convertendo colunas nominais para o tipo 'category'...")
    for col in nominais:
        if col in df_proc.columns:
            # A coluna MSSubClass é numérica mas representa uma categoria
            df_proc[col] = df_proc[col].astype('category')
        else:
            print(f"Aviso: Coluna nominal '{col}' não encontrada no DataFrame.")
            
    print("\nPré-processamento concluído.")
    return df_proc

'''

'''
