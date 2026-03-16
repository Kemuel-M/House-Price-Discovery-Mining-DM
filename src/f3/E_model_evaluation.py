import pandas as pd
import xgboost as xgb
import numpy as np

def gerar_submissao_kaggle(modelo_treinado: xgb.XGBRegressor, df_teste: pd.DataFrame):
    """
    Usa um modelo XGBoost treinado para fazer previsões em um conjunto de teste,
    formata os resultados e os salva em um arquivo CSV para submissão no Kaggle.

    A função assume que o modelo foi treinado em um alvo transformado por np.log1p()
    e reverte essa transformação para a submissão final.

    Args:
        modelo_treinado (xgb.XGBRegressor): O modelo XGBoost já treinado.
        df_teste (pd.DataFrame): O DataFrame de teste, que não contém a coluna 'SalePrice'.
    """
    # 1. Fazer uma cópia para evitar alterar o DataFrame original
    df_teste_proc = df_teste.copy()

    # 2. Salvar a coluna 'Id' para o arquivo final e preparar o X_teste
    ids_teste = df_teste_proc['Id']
    X_teste = df_teste_proc.drop('Id', axis=1)

    # 3. Garantir que as colunas categóricas do teste tenham o mesmo tipo que no treino
    # Isso é CRUCIAL para que o XGBoost com 'enable_categorical=True' funcione corretamente.
    print("Preparando dados de teste...")
    for col in X_teste.select_dtypes(include=['object', 'category']).columns:
        X_teste[col] = X_teste[col].astype('category')

    # 4. Fazer as previsões (elas sairão na escala de log, pois o modelo foi treinado assim)
    print("Gerando previsões...")
    previsoes_log = modelo_treinado.predict(X_teste)

    # 5. Reverter a transformação logarítmica para obter o preço real
    # O modelo previu log(1+SalePrice), então usamos exp(x) - 1 para reverter.
    previsoes_preco_final = np.expm1(previsoes_log)
    
    # Garantir que não há previsões negativas (um efeito colateral raro, mas possível)
    previsoes_preco_final[previsoes_preco_final < 0] = 0

    # 6. Criar o DataFrame de submissão no formato exigido
    df_submissao = pd.DataFrame({
        'Id': ids_teste,
        'SalePrice': previsoes_preco_final
    })

    # 7. Salvar o DataFrame em um arquivo .csv
    nome_arquivo = 'submission.csv'
    df_submissao.to_csv(nome_arquivo, index=False)

    print("-" * 50)
    print(f"Arquivo de submissão '{nome_arquivo}' criado com sucesso!")
    print(f"O arquivo contém {len(df_submissao)} previsões.")
    print("\nExemplo das primeiras 5 linhas do arquivo gerado:")
    print(df_submissao.head())
    print("-" * 50)