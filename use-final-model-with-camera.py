import numpy as np
import onnxruntime as ort
import cv2 as cv
import time

session = ort.InferenceSession("dino_deits8-480-final.onnx", providers=['CUDAExecutionProvider'])
input_names = [input.name for input in session.get_inputs()]
output_names = [output.name for output in session.get_outputs()]

camera = cv.VideoCapture(0,cv.CAP_DSHOW) 

t0 = int(time.time())
i = 0
fps = 0
while True:
    hasFrame, frame = camera.read()
    if not hasFrame:
        break
    i += 1
    
    image_size = (480, 480)
    blob = cv.dnn.blobFromImage(frame, 1.0/255, image_size, (0, 0, 0), swapRB=True, crop=True)
    blob[0][0] = (blob[0][0] - 0.485)/0.229
    blob[0][1] = (blob[0][1] - 0.456)/0.224
    blob[0][2] = (blob[0][2] - 0.406)/0.225
            
    data_input = { input_names[0] : blob }
    data_output = session.run(output_names, data_input)
    features = data_output[0][0]
    attentions = data_output[1][0]

    nh = attentions.shape[0]
    attentions = attentions[:, 0, 1:].reshape(nh, -1)
    patch_size = 8
    w_featmap, h_featmap = image_size[0] // patch_size, image_size[1] // patch_size
    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    maps = []
    for j in range(nh):
        attention = attentions[j]
        attention /= np.max(attention)
        attention = np.asarray(attention*255,np.uint8)
        attention = cv.resize(attention,image_size,interpolation=cv.INTER_NEAREST)
        maps.append(attention)

    disp = cv.vconcat([cv.hconcat(maps[:nh//2]),cv.hconcat(maps[nh//2:])])
    cv.imshow('attention maps',disp)
    t1 = int(time.time())
    if t1 != t0:
        fps = i
        i = 0
        t0 = t1
    if fps > 0:
        cv.putText(frame, str(fps), (50, 50), 0, 1.0, (0, 0, 255), 2)
    cv.imshow('camera',frame)
    key = cv.waitKey(1)
    if key == 27:
        break
    elif key == ord('s'):
        cv.imwrite(str(int(time.time()*10))+'.png',disp)

cv.destroyAllWindows()