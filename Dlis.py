import dlisio

# Caminho do arquivo .dlis
caminho_arquivo = '/C:/caminho/do/arquivo.dlis'

# Abre o arquivo .dlis
with dlisio.load(caminho_arquivo) as arquivo_dlis:
    # Itera sobre os registros do arquivo
    for registro in arquivo_dlis:
        # Itera sobre os canais do registro
        for canal in registro.channels:
            # Imprime os dados do canal
            print(f'Dados do canal {canal.name}:')
            for dado in canal:
                print(dado)