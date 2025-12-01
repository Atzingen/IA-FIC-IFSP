from transformers import Sam3Processor, Sam3Model
from PIL import Image
import torch
import numpy as np
import random
import sys

# Ajuste para o seu caso
IMAGE_PATH = "chess-photos/photo_20251130_135227.jpg"
TEXT_PROMPT = "chess piece"  # ex.: "a person", "a dog", "yellow school bus"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Carregar modelo e processor do Hugging Face Hub
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Carregar imagem
image = Image.open(IMAGE_PATH).convert("RGB")
# base_np será usado para todos os overlays
base_np = np.array(image).astype("float32")

# Preparar inputs com prompt de texto
inputs = processor(
    images=image,
    text=TEXT_PROMPT,
    return_tensors="pt",
)

# Enviar tudo para o device (cuda/cpu)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Pós-processamento: masks/boxes/scores no tamanho original
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs["original_sizes"].tolist(),
)[0]

masks = results["masks"]    # [num_instances, H, W], bool
boxes = results["boxes"]    # [num_instances, 4] em pixels xyxy
scores = results["scores"]  # [num_instances]

num_instances = len(masks)
print(f"Prompt: {TEXT_PROMPT}")
print(f"Objetos encontrados: {num_instances}")

if num_instances == 0:
    print("Nenhuma máscara encontrada. Ajuste o texto ou thresholds.")
    sys.exit(0)

alpha = 0.6  # transparência (0: sem cor, 1: cor sólida)

# Imagem combinada com todos os objetos
overlay_all_np = base_np.copy()

for idx, (mask_tensor, box, score) in enumerate(zip(masks, boxes, scores)):
    mask = mask_tensor.cpu().numpy().astype(bool)
    if not mask.any():
        continue

    # Cor aleatória (RGB) para este objeto
    color = np.array(
        [random.randint(0, 255) for _ in range(3)],
        dtype="float32"
    )

    print(
        f"Obj {idx}: score={float(score):.3f}, "
        f"box={box.tolist()}, color={color.tolist()}"
    )

    # Aplica cor com transparência onde mask == True
    overlay_all_np[mask] = (
        overlay_all_np[mask] * (1.0 - alpha) + color * alpha
    )

# Converte e salva imagem final
overlay_all_img = Image.fromarray(
    overlay_all_np.clip(0, 255).astype("uint8")
)
overlay_all_img.save("overlay_all.png")

print("Imagem única gerada: overlay_all.png")