from skl2onnx.helpers.onnx_helper import load_onnx_model
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
#http://onnx.ai/sklearn-onnx/auto_examples/plot_intermediate_outputs.html#compute-intermediate-outputs

model_onnx = load_onnx_model("dino_deits8-480.onnx")
outnames = [out for out in enumerate_model_node_outputs(model_onnx)]

#outname = '1260' # (785, 384)
#num_onnx = select_model_inputs_outputs(model_onnx, outname)
#_ = save_onnx_model(num_onnx, "model1260.onnx")

#outname = ['1260','output'] # (785, 384) a (384,)
#num_onnx = select_model_inputs_outputs(model_onnx, outname)
#_ = save_onnx_model(num_onnx, "model1260output.onnx")

num_onnx = select_model_inputs_outputs(model_onnx, outnames)
_ = save_onnx_model(num_onnx, "dino_deits8-480-all-outputs.onnx")

