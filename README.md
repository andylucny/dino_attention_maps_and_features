# dino_attention_maps_and_features
export of the dino model providing both attention maps and the feature vector of the attended object to ONNX

This projects employs code and model from https://github.com/facebookresearch/dino

All credits for quality of the model belong to the MicroSoft Research.

We convert the model to ONNX and manipulate it to get both attentions maps and the feature vector.

A the first we run the original model, patched to support a fluent conversion to ONNX in directory dino via run.bat
It downloads the model, processes the img.png to six attention maps in the upper directory and also saves 
the complete model there as dino_deits8-480.onnx

Further we need to found name of the intermediate result corresponding to the attention maps. 
So we change the model to output all intermediate results by intermediate-model.py that saves dino_deits8-480-all-outputs.onnx.

Then we investigate shape and first values of each intermediate result by intermediate-outputs.py. 
We find that the feature vector is named e.g. '1192' and the attentions maps '/blocks.11/attn/Softmax_output_0'.

By final-model.py we create the model dino_deits8-480-final.onnx which output exactly these two results.

Finnally we use the model to get both attention maps and the feature vector for the image img.png running use-final-model.py. 
While the attentions maps tell us where the bird is on the image, the feature vector enables us to distinguish the bird for other objects.

The final model is available at https://www.agentspace.org/download/dino_deits8-480-final.onnx for input 480x480
and https://www.agentspace.org/download/dino_deits8-224-final.onnx for input 224x224.

It is possible to download it directly via download-final-model.py (and then use only use-final-model.py)

Microsoft Research licenced this model under Apache 2.0 licence.
