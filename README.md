# Segmentação Semântica – U-Net + TFLite (FP32)

## Dados e Modelo
- Dataset: Oxford-IIIT Pet (torchvision)
- Subset reduzido para treino rápido
- Modelo: U-Net (saída binária pet vs não-pet)

## Métricas (val)
- IoU: 0.8885
- Acurácia: 0.890
- Exemplos em `training/samples/`

## Treino (Python 3.10)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # (ou /cpu)
pip install -r training/requirements.txt
python training/train_unet_pets.py

## Conversão (Colab)
- Upload de `unet_best.pt`
- ONNX -> TF -> TFLite FP32 → gera `model_float32.tflite`

## App Android
- Copiar `model_float32.tflite` para `app/src/main/assets/`
- Build & Run ▶
- Selecionar imagem → Segmentar (U-Net TFLite)

