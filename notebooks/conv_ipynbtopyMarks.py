#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json  # Trocamos nbformat por json

def convert_notebook_to_py(notebook_path, output_path=None):
    """
    Converte um Jupyter notebook para um script Python usando apenas a biblioteca json padrão.
    """
    # Se output_path não for fornecido, usa o mesmo nome com extensão .py
    if output_path is None:
        output_path = os.path.splitext(notebook_path)[0] + '.py'

    try:
        # Lê o notebook como um arquivo JSON comum
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Abre o arquivo de saída para escrita
        with open(output_path, 'w', encoding='utf-8') as f:

            # Verifica se a chave 'cells' existe (estrutura básica do ipynb)
            if 'cells' not in notebook:
                print("Erro: Estrutura do arquivo inválida (chave 'cells' ausente).")
                return False

            # Processa cada célula
            for cell in notebook['cells']:
                cell_type = cell.get('cell_type', '')

                # O campo 'source' no JSON bruto geralmente é uma lista de strings.
                # Juntamos tudo para tratar como texto único.
                source_content = cell.get('source', [])
                if isinstance(source_content, list):
                    source_text = ''.join(source_content)
                else:
                    source_text = str(source_content)

                if cell_type == 'markdown':
                    # Processa célula markdown
                    f.write("# %% [markdown]\n")
                    # Converte cada linha de markdown para um comentário Python
                    for line in source_text.splitlines():
                        f.write(f"# {line}\n")
                    f.write("\n")

                elif cell_type == 'code':
                    # Processa célula de código
                    f.write("# %%\n")
                    f.write(source_text)
                    f.write("\n\n")

        print(f"Conversão concluída: {notebook_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Erro durante a conversão: {e}")
        return False

def get_valid_notebook_path():
    """Obtém um caminho válido de notebook do usuário."""
    while True:
        filename = input("Digite o nome do arquivo Jupyter notebook: ")
        if not filename:
            continue

        notebook_path = filename if filename.endswith('.ipynb') else filename + '.ipynb'

        if os.path.exists(notebook_path):
            return notebook_path
        else:
            print(f"Arquivo '{notebook_path}' não encontrado. Por favor, tente novamente.")

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        notebook_path = filename if filename.endswith('.ipynb') else filename + '.ipynb'

        if not os.path.exists(notebook_path):
            print(f"Arquivo '{notebook_path}' não encontrado.")
            notebook_path = get_valid_notebook_path()
    else:
        notebook_path = get_valid_notebook_path()

    convert_notebook_to_py(notebook_path)

if __name__ == "__main__":
    main()
