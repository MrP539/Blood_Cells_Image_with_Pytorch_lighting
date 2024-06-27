import torch
import torchvision
import os
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix
from create_confusion_matrix import CREATE_CONFUSION_MATRICS

# Define the transformation for the test set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test set
test_set = torchvision.datasets.ImageFolder(os.path.join("data_set", "test"), transform=transform)

# Define the LightningModule for Bloodcell
class Bloodcell(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = torchvision.models.resnet50(pretrained=False)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, self.n_classes)
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.backbone(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        #loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=self.n_classes)
        #self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        self.test_preds.append(preds)
        self.test_targets.append(y)
        return {'preds': preds, 'targets': y}

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        print(classification_report(targets.cpu(), preds.cpu(), target_names=test_set.classes))
        print("Confusion Matrix:\n", confusion_matrix(targets.cpu(), preds.cpu()))
        CREATE_CONFUSION_MATRICS(y_actual=targets.cpu(),y_pred=preds.cpu())

        

# Load the model from checkpoint
checkpoint_path = r"D:\machine_learning_AI_Builders\บท4\Classification\Blood_Cells_Image_Dataset_pytorch_lighting\result_model\ResNet50_Adam\resnet50--epoch=42-val_accuracy=0.86-val_loss=0.42_Adam.ckpt"  # Replace with your checkpoint path
model = Bloodcell.load_from_checkpoint(checkpoint_path=os.path.join(checkpoint_path), n_classes=len(test_set.classes))
# Test the model
trainer = pl.Trainer()
trainer.test(model, dataloaders=torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False))

