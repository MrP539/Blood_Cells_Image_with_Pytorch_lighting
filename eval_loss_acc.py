import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\Classification\Blood_Cells_Image_Dataset_pytorch_lighting\csv_result\version_0\metrics.csv")
df = df.groupby(['epoch', 'step']).last().reset_index()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.train_loss,label= "train_loss",color = "blue",marker ="x")
plt.plot(df.epoch,df.val_loss,label= "val_loss",color = "red",marker ="x")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss-epoch")
plt.show()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.val_accuracy,label= "accuracy",color = "blue",marker ="x")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("accuracy")
plt.show()
