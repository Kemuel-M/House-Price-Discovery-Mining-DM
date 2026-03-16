'''
remover a coluna 'Id' do DataFrame, pois não é uma feature útil para o modelo.
Criar um modelo de regressão XGBoost para prever o atributo 'SalePrice'.
    deve ter (enable_categorical=True, tree_method='hist'), pois está preparado para lidar com dados categóricos.
Use validação cruzada com 10 folds na avaliação do modelo XGBoost.
    Avalie o modelo usando R², RMSE e MAE.
    Gere gráficos de dispersão entre os valores reais e previstos, além de histogramas dos resíduos.
    Implemente a otimização de hiperparâmetros usando GridSearchCV para melhorar o desempenho do modelo.
    Considere ajustar os seguintes hiperparâmetros:
        objective: ['reg:squarederror', 'reg:linear']
        min_child_weight: [1, 2]
        gamma: [0.5, 1]
        subsample: [0.2, 0.4]
        max_depth: [1, 2]
'''

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_validate, train_test_split
from xgboost.callback import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def treinar_e_avaliar_xgboost(df: pd.DataFrame):
    """
    Remove a coluna 'Id', treina um modelo XGBoost para prever 'SalePrice'
    e o avalia usando validação cruzada com 10 folds.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados de casas.

    Returns:
        xgb.XGBRegressor: O modelo XGBoost treinado.
    """
    # 1. Preparar os dados
    # Remover a coluna 'Id'
    df = df.drop('Id', axis=1)
    # Separar as features (X) e o alvo (y)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    # Análise e Transformação da variável alvo 'SalePrice'
    # Aplicar log1p (log(1+x)) para normalizar a distribuição
    y_log = np.log1p(y)
    # Garantir que as colunas categóricas estão no formato correto para o XGBoost
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category')

    # 2. Criar um modelo APENAS para a validação cruzada (sem callbacks)
    modelo_para_cv = xgb.XGBRegressor(
        objective='reg:squarederror',
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        eval_metric='rmse',
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.005
    )

    # 3. Usar validação cruzada com 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 4. Avaliar o modelo usando R², RMSE e MAE
    # Como a variável alvo foi transformada, as métricas de erro (RMSE, MAE)
    # serão na escala de log. O R² não é afetado da mesma forma.
    scoring = {
        'r2': 'r2',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error'
    }

    # Executar a validação cruzada
    scores = cross_validate(modelo_para_cv, X, y_log, cv=kf, scoring=scoring, n_jobs=-1)

    # Calcular as médias e desvios padrão das métricas
    mean_r2 = np.mean(scores['test_r2'])
    std_r2 = np.std(scores['test_r2'])
    mean_rmse_log = -np.mean(scores['test_neg_root_mean_squared_error'])
    std_rmse_log = np.std(scores['test_neg_root_mean_squared_error'])
    mean_mae_log = -np.mean(scores['test_neg_mean_absolute_error'])
    std_mae_log = np.std(scores['test_neg_mean_absolute_error'])

    # Exibir os resultados (erros na escala de log)
    print("Resultados da Avaliação com Validação Cruzada (10 folds) - Alvo em Escala de Log:")
    print("-" * 70)
    print(f"R² (R-squared):                   {mean_r2:.4f} (+/- {std_r2:.4f})")
    print(f"RMSE (Root Mean Squared Error):   {mean_rmse_log:.4f} (+/- {std_rmse_log:.4f})")
    print(f"MAE (Mean Absolute Error):        {mean_mae_log:.4f} (+/- {std_mae_log:.4f})")
    print("-" * 70)

    # 5. AGORA, criar o modelo final com os callbacks para o treinamento
    modelo_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        eval_metric='rmse',
        callbacks=[EarlyStopping(rounds=50, save_best=True)], # CALLBACKS APLICADOS AQUI
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.005
    )

    # 5. Treinar o modelo final com todos os dados
    # Para usar early stopping, precisamos de um conjunto de validação
    X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.1, random_state=42)

    modelo_xgb.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False
    )

    print(f"\nModelo final treinado com {modelo_xgb.best_iteration} árvores (early stopping).")


    print("\nResultados da Avaliação do Modelo Final no Conjunto de Validação:")
    print("-" * 70)

    # Fazer previsões no conjunto de validação
    predicoes_val = modelo_xgb.predict(X_val)

    # Calcular métricas na escala de log (como o modelo foi treinado)
    r2_val = r2_score(y_val_log, predicoes_val)
    rmse_val_log = np.sqrt(mean_squared_error(y_val_log, predicoes_val))
    mae_val_log = mean_absolute_error(y_val_log, predicoes_val)

    print("Métricas na Escala de Log:")
    print(f"R² (R-squared):                   {r2_val:.4f}")
    print(f"RMSE (Root Mean Squared Error):   {rmse_val_log:.4f}")
    print(f"MAE (Mean Absolute Error):        {mae_val_log:.4f}")

    # Reverter a transformação log para métricas na escala original (mais interpretável)
    y_val_original = np.expm1(y_val_log)
    predicoes_val_original = np.expm1(predicoes_val)

    rmse_val_original = np.sqrt(mean_squared_error(y_val_original, predicoes_val_original))
    mae_val_original = mean_absolute_error(y_val_original, predicoes_val_original)
    
    print("\nMétricas na Escala Original de Preço ($):")
    print(f"RMSE (Root Mean Squared Error):   ${rmse_val_original:,.2f}")
    print(f"MAE (Mean Absolute Error):        ${mae_val_original:,.2f}")
    print("-" * 70)
    print(f"(Isso significa que, em média, as previsões de preço do modelo final erram por aproximadamente ${mae_val_original:,.2f})")


    # Extrair e plotar a importância das features
    feature_importances = pd.DataFrame(
        modelo_xgb.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    # Plotar as 15 features mais importantes
    plt.figure(figsize=(10, 8))
    sb.barplot(x=feature_importances.head(15).importance, y=feature_importances.head(15).index)
    plt.title('Importância das Features')
    plt.show()

    return modelo_xgb

'''
# melhorias:
Análise e Transformação da variável alvo 'SalePrice'
frequentemente têm uma distribuição assimétrica (com uma "cauda" longa à direita), e a transformação de log ajuda a normalizar essa distribuição.
divide essa funcao em tipos de treino. e o plot de importancia das features em outra funcao. deixa a parte de dividir os dados dentro de cada treino, ai fica a divisao especifica de cada modelo.
comparar a importancia de features entre o que o modelo disse e a análise de correlação.
o ano por exemplo, pode mudar de numerico, para categorico de de 10 em 10 anos, e ver se melhora o modelo.
usar regras de associação para criar novas features talvez, ou outras tecnicas de feature engineering. ou usar essas regras de outra forma para prever os preços.
plotar a escala linear dos erros, para ver a distribuicao dos erros.
Ver os 100 maiores erros, e ver se tem algum padrao.
hisGB_ord pro colega foi o melhor modelo que ele testou, talvez testar ele aqui tambem.
'''