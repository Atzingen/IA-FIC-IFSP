#!/usr/bin/env bash
set -euo pipefail

# Sempre começa no diretório onde está este script
cd "$(dirname "$0")"

# Clona o repositório se ainda não existir
if [ ! -d "text-datasets" ]; then
  git clone https://github.com/falabrasil/text-datasets.git
fi

cd text-datasets

echo "==> Baixando BlogSet-BR em datasets/blogset ..."

mkdir -p datasets/blogset
chmod +x src/fetch/get_blogset.sh
src/fetch/get_blogset.sh datasets/blogset

cd ..

echo "==> Rodando extract_blogset_to_txt.py ..."
python extract_blogset_to_txt.py

echo "==> Pronto."
wc -l input.txt || true
du -h input.txt || true
