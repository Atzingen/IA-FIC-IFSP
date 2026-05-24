# IA-FIC-IFSP

Material do curso FIC **"Introdução à IA para Tomada de Decisão"** — IFSP Piracicaba, 2021/2 em diante. Repo é **scaffolding pedagógico**, não código de produção.

## Estrutura

- 12 aulas (`Aula_00` … `Aula_12`) + 3 atividades (`Atividade_0x`), cada uma autoexplicativa.
- Cada `Aula_*/README.md` linka aulas no YouTube (vídeo é a fonte primária; código é companion).
- Mix de `.ipynb` (Jupyter) e `.py` standalone.

## Stack

Python 3.x + pandas, numpy, scikit-learn, TensorFlow/Keras, OpenCV. Pontual: Gradio (UI), LangChain (Aula_11 RAG, precisa `OPENAI_API_KEY` em `.env`), SAM v2 + YOLO11 (Aula_12).

- **Sem `requirements.txt` central** — cada aula assume libs pré-instaladas no env. Para reproduzir sem dor, usar `conda env` por aula quando alguma versão briga.
- **Pesos de modelo grandes ignorados** (`.pt`, `.onnx`, ~450 MB SAM/YOLO). Baixar conforme a aula.
- **Gabaritos ignorados** no `.gitignore` (`*gabarito*`).

## Como trabalhar aqui

- Aulas têm dependência temática: Aula_00 (Python básico) → Aula_12 (CV + LLM). Não assumir conceito de aula posterior.
- README em PT-BR; código mistura PT/EN nos comentários.
- Mudanças em material didático: priorizar **clareza > performance**, pseudocódigo onde fizer sentido.
- Não há test suite — verificação é rodar o notebook end-to-end.

## Env

Neste notebook: conda env `base` (Python 3.13) costuma servir. Para CV/LLM (Aulas 11–12), criar env dedicado para evitar conflito de versões.
