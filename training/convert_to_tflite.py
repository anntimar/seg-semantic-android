# Conversão: PyTorch (checkpoint) -> ONNX -> TF SavedModel -> TFLite (FP32)
# Requer: onnx, onnxsim, onnx2tf, tensorflow
import os, torch
from pathlib import Path
import onnx
from onnxsim import simplify
from onnx2tf import convert

# importar UNet da mesma pasta de treino
from train_unet_pets import UNet, DEVICE, IMG_SIZE

CHK = "training/unet_best.pt"
onnx_raw = "training/unet.onnx"
onnx_simp = "training/unet_simp.onnx"
out_dir = "training/unet_tf"

def main():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(CHK, map_location=DEVICE))
    model.eval()

    dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE, device=DEVICE)
    Path("training").mkdir(exist_ok=True, parents=True)
    torch.onnx.export(
        model, dummy, onnx_raw,
        input_names=["input"], output_names=["logits"],
        opset_version=17, do_constant_folding=True
    )
    # simplificar
    m = onnx.load(onnx_raw)
    m_simp, ok = simplify(m)
    assert ok, "Falha ao simplificar ONNX"
    onnx.save(m_simp, onnx_simp)

    # converter p/ TF + TFLite (FP32)
    convert(
        input_onnx_file_path=onnx_simp,
        output_folder_path=out_dir,
        output_integer_quantized_tflite=False,
        copy_onnx_input_output_names_to_tflite=True,
    )
    print("OK ->", out_dir, "(model_float32.tflite)")

if __name__ == "__main__":
    main()
