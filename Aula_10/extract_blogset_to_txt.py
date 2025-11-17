import gzip
import csv
import os
import sys

CSV_GZ_PATH = "text-datasets/datasets/blogset/blogset-br.csv.gz"
OUT_PATH = "input.txt"

if not os.path.exists(CSV_GZ_PATH):
    sys.stderr.write(f"ERRO: arquivo não encontrado: {CSV_GZ_PATH}\n")
    sys.exit(1)

# Aumenta limite de tamanho de campo (linhas muito grandes)
csv.field_size_limit(sys.maxsize)

linhas_lidas = 0
posts_escritos = 0

def limpar_texto_bruto(texto: str) -> str:
    """
    Remove linhas vazias, linhas só com &nbsp/espacos,
    normaliza quebra de linha.
    """
    # Normaliza alguns caracteres
    texto = texto.replace("\r", "\n")
    texto = texto.replace("\u00a0", " ")  # NBSP
    texto = texto.replace("&nbsp;", " ")
    texto = texto.replace("&nbsp", " ")

    linhas = texto.splitlines()
    linhas_limpa = []

    for linha in linhas:
        # Normaliza para checar se está vazia
        normalizada = linha.strip()
        normalizada = normalizada.replace("\u00a0", " ").replace("&nbsp;", " ").replace("&nbsp", " ").strip()
        if not normalizada:
            # pula linha vazia / só com espaços/&nbsp
            continue
        linhas_limpa.append(linha.strip())

    if not linhas_limpa:
        return ""

    # junta de volta com uma quebra de linha simples entre linhas
    return "\n".join(linhas_limpa)


with gzip.open(CSV_GZ_PATH, mode="rt", encoding="utf-8", newline="") as f_in, \
     open(OUT_PATH, mode="w", encoding="utf-8") as f_out:

    reader = csv.reader(f_in, delimiter=';')

    # Lê primeira linha para tentar identificar cabeçalho
    first = next(reader, None)
    if first is None:
        sys.stderr.write("ERRO: CSV vazio.\n")
        sys.exit(1)

    header_lower = [c.strip().lower() for c in first]
    if "content" in header_lower:
        content_idx = header_lower.index("content")
        # 'first' é cabeçalho, não é dado
    else:
        # sem cabeçalho: assume coluna 4 (0-based)
        content_idx = 4
        linhas_lidas += 1
        if len(first) > content_idx:
            bruto = first[content_idx]
            texto_limpo = limpar_texto_bruto(bruto)
            if texto_limpo:
                f_out.write(texto_limpo + "\n\n")  # separa posts com linha em branco
                posts_escritos += 1

    for row in reader:
        linhas_lidas += 1
        if len(row) <= content_idx:
            continue
        bruto = row[content_idx]
        texto_limpo = limpar_texto_bruto(bruto)
        if not texto_limpo:
            continue
        f_out.write(texto_limpo + "\n\n")
        posts_escritos += 1

print(f"Lidas {linhas_lidas} linhas de dados.")
print(f"Gravados {posts_escritos} posts em {OUT_PATH}.")
