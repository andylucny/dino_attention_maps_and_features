print('please, wait...')
import requests
import os

def download_and_save(url,path):
    if os.path.exists(path):
        return
    print("downloading",path)
    response = requests.get(url)
    open(path,"wb").write(response.content)
    print(path,"downloaded")
    
def download_DinoViT_model():
    download_and_save("https://www.agentspace.org/download/dino_deits8-480-final.onnx","dino_deits8-480-final.onnx")

def download_all():
    download_DinoViT_model()

if __name__ == "__main__":
    download_all()
    print("done")