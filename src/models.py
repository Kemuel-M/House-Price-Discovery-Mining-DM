import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, train_test_split
from xgboost.callback import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def treinar_e_avaliar_xgboost(df: pd.DataFrame, target: str = 'SalePrice'):
    """Treina o modelo XGBoost com validação cruzada e early stopping."""
    # Preparação
    X = df.drop(columns=['Id', target], errors='ignore')
    y = df[target]
    y_log = np.log1p(y)

    # Garantir tipos categóricos
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category')

    params = {
        'objective': 'reg:squarederror',
        'enable_categorical': True,
        'tree_method': 'hist',
        'random_state': 42,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'gamma': 0.1
    }

    # 1. Validação Cruzada (10 folds)
    print("\nExecutando Validação Cruzada (10 folds)...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    modelo_cv = xgb.XGBRegressor(**params)
    
    cv_results = cross_validate(
        modelo_cv, X, y_log, cv=kf,
        scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
        n_jobs=-1
    )

    print(f"R² Médio: {np.mean(cv_results['test_r2']):.4f}")
    print(f"RMSE Médio (log): {-np.mean(cv_results['test_neg_root_mean_squared_error']):.4f}")

    # 2. Treinamento Final com Early Stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.1, random_state=42)
    
    modelo_final = xgb.XGBRegressor(**params, callbacks=[EarlyStopping(rounds=50, save_best=True)])
    modelo_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print(f"\nModelo final treinado. Melhor iteração: {modelo_final.best_iteration}")
    
    # Previsões para validação de métricas reais
    preds_log = modelo_final.predict(X_val)
    preds_orig = np.expm1(preds_log)
    y_val_orig = np.expm1(y_val)

    mae_orig = mean_absolute_error(y_val_orig, preds_orig)
    print(f"MAE na escala original: ${mae_orig:,.2f}")

    # Exibir Top Features textualmente para análise
    importances = modelo_final.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    print("\n🔝 TOP 10 FEATURES (IMPORTÂNCIA DO MODELO):")
    print(importance_df.head(10).to_string(index=False))

    return modelo_final, list(X.columns)

def gerar_submissao(modelo, df_test: pd.DataFrame, train_columns: list, output_path: str = "submission.csv"):
    """Gera o arquivo de submissão para o Kaggle, garantindo alinhamento de colunas."""
    X_test = df_test.drop(columns=['Id'], errors='ignore')
    
    # Garantir tipos categóricos
    for col in X_test.select_dtypes(include=['object', 'category']).columns:
        X_test[col] = X_test[col].astype('category')

    # Alinhamento de colunas (Garante que o teste tenha as mesmas colunas do treino na mesma ordem)
    X_test = X_test.reindex(columns=train_columns)
    
    # Preencher possíveis NaNs introduzidos pelo reindex (colunas que existiam no treino mas não no teste)
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category':
            if X_test[col].isnull().any():
                # Se a coluna não existir no teste, ela virá toda NaN. O XGBoost lida com NaN em categorias, 
                # mas é mais seguro garantir que o tipo permaneça consistente.
                pass 
        elif X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(0)

    preds_log = modelo.predict(X_test)
    preds_final = np.expm1(preds_log)
    preds_final[preds_final < 0] = 0

    submission = pd.DataFrame({
        'Id': df_test['Id'],
        'SalePrice': preds_final
    })
    submission.to_csv(output_path, index=False)
    print(f"\n✔️ Submissão salva em {output_path}")
    return submission
