from skl2onnx.helpers.onnx_helper import load_onnx_model
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs

model_onnx = load_onnx_model("dino_deits8-480.onnx")
outnames = ['1192','/blocks.11/attn/Softmax_output_0'] #,'/blocks.11/attn/MatMul_output_0','/blocks.11/attn/Mul_output_0']

num_onnx = select_model_inputs_outputs(model_onnx, outnames)
_ = save_onnx_model(num_onnx, "dino_deits8-480-final.onnx")
