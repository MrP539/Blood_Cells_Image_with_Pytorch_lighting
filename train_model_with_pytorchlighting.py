import pytorch_lightning as pl
import sklearn.metrics
import torch
import torch.utils
import torch.utils.data
import torchvision
import os
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import sklearn
from create_confusion_matrix import CREATE_CONFUSION_MATRICS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loot_path = r"D:\machine_learning_AI_Builders\บท4\Classification\Blood_Cells_Image_with_Pytorch_lighting"

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((224,224),scale=(0.2,0.8)),
    torchvision.transforms.CenterCrop(180),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.TrivialAugmentWide(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])
])

valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageFolder(os.path.join(loot_path,"data_set","train"),transform=train_transform)
valid_set = torchvision.datasets.ImageFolder(os.path.join(loot_path,"data_set","valid"),transform=valid_transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True,num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=32,shuffle=False,num_workers=0)

class BloodCell(pl.LightningModule):
    def __init__(self,n_classes=len(train_set.classes)):
        super(BloodCell,self).__init__()

        self.n_classes  =n_classes
        self.backbone = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features,out_features=self.n_classes)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass",num_classes = self.n_classes)

    def forward(self,x):
        pred = self.backbone(x)
        return pred
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        logist = self.backbone(x)
        loss = self.loss_function(logist,y)
        y_pred = torch.argmax(logist,dim=1)
        self.log("train_loss",loss,on_step=False,on_epoch=True)

        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        logist = self.backbone(x)
        loss = self.loss_function(logist,y)
        y_pred = torch.argmax(logist,dim=1)
        self.log("val_loss",loss,on_step=False,on_epoch=True)
        self.log("val_accuracy",self.accuracy(y_pred,y),on_step=False,on_epoch=True)

        return loss
      
    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(),lr=3e-3)
        return {"optimizer":self.optimizer,
                "monitor":"val_loss"
                }
checkpoint_callback = ModelCheckpoint(
   dirpath="./checkpoints/",
   filename="resnet50--{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}_SGD",
   save_top_k=1,
   verbose=True,
   monitor="val_loss",
   mode="min",
) 

csv_logger = CSVLogger(loot_path,"csv_result")
model = BloodCell(n_classes=len(train_set.classes))
print(model)

trainer = pl.Trainer(max_epochs=1,devices=1,callbacks=[checkpoint_callback],logger=csv_logger,accelerator='gpu')
trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=valid_loader)
