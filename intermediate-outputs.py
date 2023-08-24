import onnxruntime as ort
import cv2 as cv

sz = 480 #224

model_name = f"dino_deits8-{sz}-all-outputs.onnx"

session = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])
input_names = [input.name for input in session.get_inputs()] # ['x.1']
output_names = [output.name for output in session.get_outputs()] # all

frame = cv.imread('img.png')
image_size = (sz, sz)
blob = cv.dnn.blobFromImage(frame, 1.0/255, image_size, (0, 0, 0), swapRB=True, crop=True)
blob[0][0] = (blob[0][0] - 0.485)/0.229
blob[0][1] = (blob[0][1] - 0.456)/0.224
blob[0][2] = (blob[0][2] - 0.406)/0.225
        
data_input = { input_names[0] : blob }
data_output = session.run(output_names, data_input)
for index in range(len(data_output)):
    output = data_output[index]
    first_values = output.reshape(-1)
    print(output_names[index],output.shape,first_values[:10])

