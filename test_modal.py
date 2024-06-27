import pytorch_lightning as pl
import torch
import torchvision
import os
from PIL import Image
from pytorch_lightning.utilities import model_helpers
import cv2

tranforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])
])

test_set = torchvision.datasets.ImageFolder(os.path.join("data_set","test"))

img_test_path = "test.jpg"

img = Image.open(os.path.join(img_test_path))
img_tensor = tranforms(img)
img_tensor.unsqueeze_(0)

if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()

class Bloodcell(pl.LightningModule):
    def __init__(self,n_classes = len(test_set.classes)) :
        super(Bloodcell,self).__init__()
        self.n_classes = n_classes
        self.backbone = torchvision.models.resnet50(weights = None)
        self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features,out_features=self.n_classes)

    def forward(self,x):
        pred = self.backbone(x)
        return pred

checkpoint_path = os.path.join(r"D:\machine_learning_AI_Builders\บท4\Classification\Blood_Cells_Image_Dataset_pytorch_lighting\checkpoints\resnet50--epoch=48-val_accuracy=0.80-val_loss=0.82_SGD.ckpt")
model = Bloodcell.load_from_checkpoint(n_classes=len(test_set.classes),checkpoint_path=checkpoint_path)

if torch.cuda.is_available():
    model = model.cuda()
print(model)
output = model(img_tensor)
index_class = torch.argmax(output,dim=1).cpu().numpy().item()
class_name = test_set.classes[index_class]
print(class_name)
img_cv2 = cv2.imread(img_test_path)
img_cv2 = cv2.resize(img_cv2,(300,300))
cv2.putText(img_cv2,f"{class_name}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)

cv2.imshow("test",img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()

