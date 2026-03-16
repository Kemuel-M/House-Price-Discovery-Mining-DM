#!/usr/bin/env python3
"""
Fase 2 – Mineração de Padrões Frequentes e Sequenciais
=====================================================

Este script reproduz (e substitui) o notebook da Fase 2 em formato `.py`,
incluindo:

1. Discretização dos atributos numéricos em 5 faixas:
   Muito Baixo, Baixo, Médio, Alto, Muito Alto
2. Transformação para formato transacional
3. Mineração de itens frequentes (Apriori) e regras de associação
4. Mineração de padrões sequenciais (PrefixSpan)
5. Relatórios simples no terminal

Uso básico
----------
```
python fase2.py --input dados.csv  # ou .pkl
```

Dependências:
-------------
- pandas
- numpy
- mlxtend
- prefixspan
- matplotlib (opcional – apenas se você inserir gráficos)

Instale via:
```
pip install pandas numpy mlxtend prefixspan matplotlib
```
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

# Tentativa de import dos pacotes de mineração – aborta com mensagem legível
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    from prefixspan import PrefixSpan
except ImportError as e:
    sys.exit(
        f"Pacotes necessários não encontrados: {e}. "
        "Instale-os com `pip install mlxtend prefixspan` e execute novamente."
    )


# --------------------------------------------------------------------------- #
# Configurações                                                               #
# --------------------------------------------------------------------------- #

CORR_COLS = [
    "OverallQual",
    "Neighborhood",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "BsmtQual",
    "KitchenQual",
    "1stFlrSF",
    "ExterQual",
    "GarageYrBlt",
    "MSSubClass",
    "FullBath",
    "YearRemodAdd",
    "LotFrontage",
    "TotRmsAbvGrd",
    "GarageType",
    "2ndFlrSF",
    "FireplaceQu",
]

BIN_LABELS = ["Muito Baixo", "Baixo", "Médio", "Alto", "Muito Alto"]


# --------------------------------------------------------------------------- #
# Funções utilitárias                                                         #
# --------------------------------------------------------------------------- #
def load_dataframe(path: pathlib.Path | None) -> pd.DataFrame:
    """
    Carrega o DataFrame pré‑processado da Fase 1.

    Se `path` for None, tenta `processed_phase1.pkl`.
    """
    if path is None:
        default = pathlib.Path("processed_phase1.pkl")
        if default.exists():
            print("✔️  Carregando", default)
            return pd.read_pickle(default)
        else:
            sys.exit(
                "Arquivo padrão 'processed_phase1.pkl' não encontrado. "
                "Use --input para indicar o caminho do dataset."
            )

    path = pathlib.Path(path)
    if not path.exists():
        sys.exit(f"Arquivo '{path}' não encontrado.")
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    elif path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    else:
        sys.exit("Formato de arquivo não suportado. Use .pkl ou .csv")


def discretize_numeric(df: pd.DataFrame, target: str = "SalePrice") -> pd.DataFrame:
    """
    Discretiza todos os atributos numéricos (exceto o alvo) em 5 quintis.
    Usa `qcut`; se falhar por falta de quantis únicos, cai para `cut`.

    Mantém atributos não‑numéricos (ou já discretizados) intocados.
    """
    df = df.copy()
    # Discretiza o alvo
    df[f"{target}Bin"] = pd.qcut(
        df[target], q=5, labels=BIN_LABELS, duplicates="drop"
    )

    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]  # exclui o alvo bruto

    for col in numeric_cols:
        try:
            df[col] = pd.qcut(df[col], q=5, labels=BIN_LABELS, duplicates="drop")
        except ValueError:
            # Fallback para cut com faixas iguais
            df[col] = pd.cut(
                df[col],
                bins=5,
                labels=BIN_LABELS,
                include_lowest=True,
                right=True,
            )

    return df


def dataframe_to_transactions(df: pd.DataFrame) -> list[list[str]]:
    """
    Converte DataFrame discreto em lista de transações (listas de strings).
    Formato 'Coluna=Valor'.
    """
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str).map(lambda v: f"{col}={v}")
    return df_str.values.tolist()


def mine_association_rules(transactions, min_sup=0.05, min_conf=0.6):
    """
    Gera itens frequentes e regras de associação.
    Retorna (freq_itemsets, rules_all, rules_target).
    """
    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_arr, columns=te.columns_)

    freq = apriori(trans_df, min_support=min_sup, use_colnames=True, max_len=5)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

    rules_target = rules[
        rules["consequents"].apply(
            lambda x: any(item.startswith("SalePriceBin=") for item in x)
        )
    ]

    return freq, rules, rules_target


def mine_sequential_patterns(sequences, min_support_ratio=0.05):
    """
    Minera padrões sequenciais usando PrefixSpan.
    """
    ps = PrefixSpan(sequences)
    min_support_abs = max(1, int(min_support_ratio * len(sequences)))
    patterns = ps.frequent(min_support_abs)
    # Ordena por comprimento desc, depois suporte desc
    patterns_sorted = sorted(patterns, key=lambda x: (-len(x[1]), -x[0]))
    return patterns_sorted


def main(args):
    df_raw = load_dataframe(args.input)
    df_rel = df_raw[CORR_COLS + ["SalePrice"]].copy()

    # Discretização
    df_disc = discretize_numeric(df_rel, target="SalePrice")

    # (Opcional) salvar csv com variáveis discretizadas
    if args.save_disc:
        outfile = pathlib.Path(args.save_disc)
        df_disc.to_csv(outfile, index=False)
        print("📁 DataFrame discretizado salvo em", outfile)

    # Transações
    transactions = dataframe_to_transactions(df_disc.drop(columns=["SalePrice"]))

    # Mineração de associação
    freq, rules, rules_target = mine_association_rules(
        transactions, min_sup=args.min_sup, min_conf=args.min_conf
    )

    # Relatórios no terminal
    print("\n=== Top 10 Itemsets Frequentes ===")
    print(freq.sort_values("support", ascending=False).head(10).to_string(index=False))

    print("\n=== Top 10 Regras (Consequente = SalePriceBin) ===")
    if not rules_target.empty:
        print(
            rules_target.sort_values("lift", ascending=False)
            .head(10)
            .loc[:, ["antecedents", "consequents", "support", "confidence", "lift"]]
            .to_string(index=False)
        )
    else:
        print("Nenhuma regra encontrada com os parâmetros fornecidos.")

    # Mineração sequencial
    attr_order = sorted(df_disc.drop(columns=["SalePrice"]).columns)
    sequences = df_disc[attr_order].astype(str).values.tolist()
    seq_patterns = mine_sequential_patterns(sequences, min_support_ratio=args.min_sup)

    print("\n=== Top 10 Padrões Sequenciais ===")
    for supp, pat in seq_patterns[:10]:
        print(f"suporte={supp:>4}  padrão={pat}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 2 – Mineração de Dados")
    parser.add_argument(
        "--input",
        help="Arquivo .pkl ou .csv com o DataFrame pré‑processado (fase1)",
        default=None,
    )
    parser.add_argument(
        "--min_sup",
        type=float,
        default=0.05,
        help="Suporte mínimo para itens frequentes (ex.: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.6,
        help="Confiança mínima para regras de associação",
    )
    parser.add_argument(
        "--save_disc",
        help="Se informado, salva o DataFrame discretizado no caminho indicado (.csv)",
    )

    args = parser.parse_args()
    main(args)
