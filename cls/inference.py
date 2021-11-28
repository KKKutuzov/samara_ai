import numpy as np

from model import Classifier
from utils import get_inference_transform



if __name__ == "__main__":
    # init
    path_to_weights = None  # soon
    transform = get_inference_transform()
    model = Classifier(path_to_weights)
    device = 'cpu'
    
    # inference
    img = np.random.randint(0, 256, (512, 512, 3))
    torch_img = transform(img).to(device).unsqueeze(0)

    clss = int(model(torch_img).argmax(-1) + 1)
    print(clss)


