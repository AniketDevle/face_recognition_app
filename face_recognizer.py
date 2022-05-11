#face_recognizer.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import build_custom_model
import io
import pickle

def get_prediction(image_bytes):
     labels_dir = "./checkpoint/labels.json"
     model_path = "./checkpoint/model_vggface2_best.pth"

     # read labels
     with open(labels_dir) as f:
          labels = json.load(f)
     print(f"labels: {labels}")

     device = torch.device('cpu')


     model = {}
     with open('custom_model_build.pickle', 'rb') as handle:
          model = pickle.load(handle)
     # model = build_custom_model.build_model(len(labels)).to(device)
     #with open('custom_model_build.pickle', 'wb') as handle:
     #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
     model.eval()

     print(f"Best accuracy of the loaded model: {torch.load(model_path, map_location=torch.device('cpu'))['best_acc']}")

     buf = io.BytesIO(image_bytes)
     img = Image.open(buf)
     img_tensor = transforms.ToTensor()(img).unsqueeze_(0).to(device)
     outputs = model(img_tensor)
     _, predicted = torch.max(outputs.data, 1)
     result = labels[np.array(predicted.cpu())[0]]
     # print(predicted.data, result)
     return result