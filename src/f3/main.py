from A_data_read_and_analyse import *
from B_data_preparation import *
from C_frequent_pattern_mining import *
from D_model_train import *
from E_model_evaluation import *


if __name__ == "__main__":
    df = read_data("../house-prices-data/train.csv")
    df = clean_and_impute_data(df)
    #analyse_data(df)
    #analyze_distributions(df)
    #features_importance = analyse_correlations(df)
    df_xgb = preparar_dados_para_xgboost(df)
    #analyse_data(df_xgb)
    #analyze_distributions(df_xgb)
    #features_importance = analyse_correlations(df_xgb)
    modelo_xgb = treinar_e_avaliar_xgboost(df_xgb)

    df_test = read_data("../house-prices-data/test.csv")
    df_test = clean_and_impute_data(df_test)
    df_test_xgb = preparar_dados_para_xgboost(df_test)
    gerar_submissao_kaggle(modelo_xgb, df_test_xgb)


'''
# implementar:
GridSearchCV para melhorar o modelo XGBoost
Curva ROC e AUC e PR
'''
