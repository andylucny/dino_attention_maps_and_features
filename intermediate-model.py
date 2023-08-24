from skl2onnx.helpers.onnx_helper import load_onnx_model
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
#http://onnx.ai/sklearn-onnx/auto_examples/plot_intermediate_outputs.html#compute-intermediate-outputs

sz = 480 #224

model_onnx = load_onnx_model(f"dino_deits8-{sz}.onnx")
outnames = [out for out in enumerate_model_node_outputs(model_onnx)]

num_onnx = select_model_inputs_outputs(model_onnx, outnames)
_ = save_onnx_model(num_onnx, f"dino_deits8-{sz}-all-outputs.onnx")

