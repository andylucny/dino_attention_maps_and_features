import numpy as np
import onnxruntime as ort
import cv2 as cv

session = ort.InferenceSession("dino_deits8-480-final.onnx", providers=['CUDAExecutionProvider'])
input_names = [input.name for input in session.get_inputs()]
output_names = [output.name for output in session.get_outputs()]

frame = cv.imread("img.png")

image_size = (480, 480)
blob = cv.dnn.blobFromImage(frame, 1.0/255, image_size, (0, 0, 0), swapRB=True, crop=True)
blob[0][0] = (blob[0][0] - 0.485)/0.229
blob[0][1] = (blob[0][1] - 0.456)/0.224
blob[0][2] = (blob[0][2] - 0.406)/0.225
        
data_input = { input_names[0] : blob }
data_output = session.run(output_names, data_input)
features = data_output[0][0]
attentions = data_output[1][0]

np.savetxt('dino_deits8-480-features.txt',features)
np.savetxt('dino_deits8-480-attentions.txt',attentions[:,0,1:])

nh = attentions.shape[0]
attentions = attentions[:, 0, 1:].reshape(nh, -1)
patch_size = 8
w_featmap, h_featmap = image_size[0] // patch_size, image_size[1] // patch_size
attentions = attentions.reshape(nh, w_featmap, h_featmap)

import os
import matplotlib.pyplot as plt
for j in range(nh):
    fname = os.path.join("./", "attn-head" + str(j) + ".png")
    plt.imsave(fname=fname, arr=cv.resize(attentions[j],image_size,interpolation=cv.INTER_NEAREST), format='png')
    print(f"{fname} saved.")

for j in range(nh):
    attention = attentions[j]
    attention /= np.max(attention)
    attention = np.asarray(attention*255,np.uint8)
    attention = cv.resize(attention,image_size,interpolation=cv.INTER_NEAREST)
    cv.imwrite("attn-head_" + str(j) + ".png",attention)

