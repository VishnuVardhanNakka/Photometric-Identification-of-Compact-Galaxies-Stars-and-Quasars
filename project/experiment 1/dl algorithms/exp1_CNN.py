import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import TensorDataset,random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from tqdm import tqdm

images = np.load('/home/amulya/Downloads/projectv/Dataset/Dataset/Experiment1/X_exp1.npy',allow_pickle=True)
labels = np.load('/home/amulya/Downloads/projectv/Dataset/Dataset/Experiment1/y_exp1.npy',allow_pickle=True)


print(images.shape)
print(labels.shape)
print(len(images))

encoder = LabelEncoder()
int = encoder.fit_transform(labels)
labels = int.reshape(-1)

permuted = np.transpose(images,(3,0,1,2))
reshaped = np.reshape(permuted,(5,-1))

mean_np = np.mean(reshaped, axis=1)
std_np = np.std(reshaped, axis=1)

images = torch.from_numpy(images)
labels = torch.from_numpy(labels)

mean = [0.013,0.0308,0.049,0.064,0.084]
std = [0.28,0.595,0.959,1.209,1.946]


transform = transforms.Normalize((mean,),(std,))
images_tr = transform(images)
print(images_tr.shape)
images_tr = images_tr.permute(0,3,1,2)




class CNNClassifier(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(5,64,kernel_size=(5,5),padding='same')
          self.relu = nn.ReLU()
          
          self.c1 = nn.Conv2d(64,48,kernel_size=(1,1),padding='same')
          self.relu1 = nn.ReLU()
          self.c2 = nn.Conv2d(64,48,kernel_size=(1,1),padding='same')
          self.relu2 = nn.ReLU()
          self.c3 = nn.Conv2d(64,48,kernel_size=(1,1),padding='same')
          self.relu3 = nn.ReLU()
          self.c4 = nn.Conv2d(48,64,kernel_size=(1,1),padding='same')
          self.relu4 = nn.ReLU()
          self.c5 = nn.Conv2d(48,64,kernel_size=(3,3),padding='same')
          self.relu5 = nn.ReLU()
          self.c6 = nn.Conv2d(48,64,kernel_size=(5,5),padding='same')
          self.relu6 = nn.ReLU()
          self.p1 = nn.AvgPool2d(kernel_size=(1,1))
          
          self.c7 = nn.Conv2d(240,64,kernel_size=(1,1),padding='same')
          self.relu7 = nn.ReLU()
          self.c8 = nn.Conv2d(240,64,kernel_size=(1,1),padding='same')
          self.relu8 = nn.ReLU()
          self.c9 = nn.Conv2d(240,64,kernel_size=(1,1),padding='same')
          self.relu9 = nn.ReLU()
          self.c10 = nn.Conv2d(64,92,kernel_size=(1,1),padding='same')
          self.relu10 = nn.ReLU()
          self.c11 = nn.Conv2d(64,92,kernel_size=(3,3),padding='same')
          self.relu11 = nn.ReLU()
          self.c12 = nn.Conv2d(64,92,kernel_size=(5,5),padding='same')
          self.relu12 = nn.ReLU()
          self.p2 = nn.AvgPool2d(kernel_size=(1,1))
          self.p3 = nn.AvgPool2d(kernel_size=(2,2))
          
          self.c13 = nn.Conv2d(340,92,kernel_size=(1,1),padding='same')
          self.relu13 = nn.ReLU()
          self.c14 = nn.Conv2d(340,92,kernel_size=(1,1),padding='same')
          self.relu14 = nn.ReLU()
          self.c15 = nn.Conv2d(340,92,kernel_size=(1,1),padding='same')
          self.relu15 = nn.ReLU()
          self.c16 = nn.Conv2d(92,128,kernel_size=(1,1),padding='same')
          self.relu16 = nn.ReLU()
          self.c17 = nn.Conv2d(92,128,kernel_size=(3,3),padding='same')
          self.relu17 = nn.ReLU()
          self.c18 = nn.Conv2d(92,128,kernel_size = (5,5),padding='same')
          self.relu18 = nn.ReLU()
          self.p4 = nn.AvgPool2d(kernel_size=(1,1))
          
          self.c19 = nn.Conv2d(476,92,kernel_size=(1,1),padding='same')
          self.relu19 = nn.ReLU()
          self.c20 = nn.Conv2d(476,92,kernel_size=(1,1),padding='same')
          self.relu20 = nn.ReLU()
          self.c21 = nn.Conv2d(476,92,kernel_size=(1,1),padding='same')
          self.relu21 = nn.ReLU()
          self.c22 = nn.Conv2d(92,128,kernel_size=(1,1),padding='same')
          self.relu22 = nn.ReLU()
          self.c23 = nn.Conv2d(92,128,kernel_size=(3,3),padding='same')
          self.relu23 = nn.ReLU()
          self.c24 = nn.Conv2d(92,128,kernel_size=(5,5),padding='same')
          self.relu24 = nn.ReLU()
          self.p5 = nn.AvgPool2d(kernel_size=(1,1))
          self.p6 = nn.AvgPool2d(kernel_size=(2,2))
          
          self.c25 = nn.Conv2d(476,92,kernel_size=(1,1),padding='same')
          self.relu25 = nn.ReLU()
          self.c26 = nn.Conv2d(476,92,kernel_size=(1,1),padding='same')
          self.relu26 = nn.ReLU()
          self.c27 = nn.Conv2d(476,128,kernel_size=(1,1),padding='same')
          self.relu27 = nn.ReLU()
          self.c28 = nn.Conv2d(92,128,kernel_size=(3,3),padding='same')
          self.relu28 = nn.ReLU()
          self.p7 = nn.AvgPool2d(kernel_size=(1,1))
          
          self.flatten = nn.Flatten()
          self.fc1 = nn.Linear(22272,1024)
          self.fc2 = nn.Linear(1024,1024)
          self.out = nn.Linear(1024, 3)
          self.softmax = nn.Softmax(dim=1)
      def forward(self,x):
          x = self.conv1(x)
          x = self.relu(x)
          c1 = self.c1(x)
          c1 = self.relu1(c1)
          c2 = self.c2(x)
          c2 = self.relu2(c2)
          c3 = self.c3(x)
          c3= self.relu3(c3)
          c4 = self.c4(c1)
          c4 = self.relu4(c4)
          c5 = self.c5(c1)
          c5 = self.relu5(c5)
          c6 = self.c6(c2)
          c6 = self.relu6(c6)
          p1 = self.p1(c3)
          x = torch.cat([c4,c5,c6,p1],dim=1)
          
          c7 = self.c7(x)
          c7 = self.relu7(c7)
          c8 = self.c8(x)
          c8 = self.relu8(c8)
          c9 = self.c9(x)
          c9 = self.relu9(c9)
          c10 = self.c10(c7)
          c10 = self.relu10(c10)
          c11 = self.c11(c7)
          c11 = self.relu11(c11)
          c12 = self.c12(c8)
          c12 = self.relu12(c12)
          p2 = self.p2(c9)
          x = torch.cat([c10,c11,c12,p2],dim=1)
          x = self.p3(x)
          
          c13 = self.c13(x)
          c13 = self.relu13(c13)
          c14 = self.c14(x)
          c14 = self.relu14(c14)
          c15 = self.c15(x)
          c15 = self.relu15(c15)
          c16 = self.c16(c13)
          c16 = self.relu16(c16)
          c17 = self.c17(c13)
          c17 = self.relu17(c17)
          c18 = self.c18(c14)
          c18 = self.relu18(c18)
          p4 = self.p4(c15)
          x = torch.cat([c16,c17,c18,p4],dim=1)
          
          c19 = self.c19(x)
          c19 = self.relu19(c19)
          c20 = self.c20(x)
          c20 = self.relu20(c20)
          c21 = self.c21(x)
          c21 = self.relu21(c21)
          c22 = self.c22(c19)
          c22 = self.relu22(c22)
          c23 = self.c23(c19)
          c23 = self.relu23(c23)
          c24 = self.c24(c20)
          c24 = self.relu24(c24)
          p5 = self.p5(c21)
          x = torch.cat([c22,c23,c24,p5],dim=1)
          x = self.p6(x)
          
          c25 = self.c25(x)
          c25 = self.relu25(c25)
          c26 = self.c26(x)
          c26 = self.relu26(c26)
          c27 = self.c27(x)
          c27 = self.relu27(c27)
          c28 = self.c28(c25)
          c28 = self.relu28(c28)
          p7 = self.p7(c26)
          x = torch.cat([c27,c28,p7],dim=1)
          
          x = self.flatten(x)
          x = self.fc1(x)
          x = self.fc2(x)
          x = self.out(x)
          x = self.softmax(x)
          
          return x
          
#splitting

dataset = TensorDataset(images_tr,labels)
x = 239999
y = 30000
set2,test_set= random_split(dataset,[x,y])
print("split")
p = 179999
q = 30000
train_set,val_set = random_split(set2,[p,q])
print("splitting")


batch_size=500
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1)
val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=1)
print("done")

model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
lr=0.01
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#print(summary(model,input_size=(batch_size,5,32,32)))


num_epochs=200
train_loss=[]
valid_loss = []
train_accuracy=[]
valid_accuracy=[]

for epoch in tqdm(range(num_epochs)):
    iter_loss = 0.0 
    correct=0
    iterations=0
    total=0
    model.train()
    
    for i,(img,lbl) in tqdm(enumerate(train_loader)):
        img = Variable(img.float())
        lbl = Variable(lbl.long())
        
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs,lbl)
        
        iter_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        _,predicted = torch.max(outputs,1)
        correct += (predicted==lbl).sum().item()
        total += lbl.size(0)
        iterations += 1
        
    train_loss.append((iter_loss / iterations))
    train_accuracy.append((100*correct / total))
     
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    model.eval()
     
    for i,(im,lb) in enumerate(val_loader):
         im = Variable(im.float())
         lb = Variable(lb.long())
         
         out = model(im)
         loss = criterion(out,lb)
         
         val_loss += loss.item()
         _,predicted_val = torch.max(outputs,1)
         correct_val += (predicted_val == lb).sum().item()
         total_val += lb.size(0)
         
    valid_loss.append(val_loss / len(val_loader))
    valid_accuracy.append((100*correct_val)/total_val)
     
    print('epoch %d/%d, trloss: %4f, tracc: %4f, valloss: %4f,valacc: %4f' % (epoch+1,num_epochs,train_loss[-1],train_accuracy[-1],valid_loss[-1],valid_accuracy[-1]))
     

f = plt.figure(figsize=(8,6))
plt.plot(train_accuracy,label='training accuracy')
plt.plot(valid_accuracy,label='valid accuracy')
plt.legend()
plt.show()



f = plt.figure(figsize=(8,6))
plt.plot(train_loss,label='training loss')
plt.plot(valid_loss,label='valid loss')
plt.legend()
plt.show()
  
X_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).float()
y_test = torch.stack([test_set[i][1] for j in range(len(test_set))]).long()
model.eval()
y_test = y_test.numpy()

y_pred = model(X_test)
y_pred = y_pred.detach().numpy()
y_pred = np.argmax(y_pred,axis=1)
print(y_pred.shape)
print(y_test.shape)

acc = accuracy_score(y_test,y_pred)
pr = precision_score(y_test,y_pred, average='macro')
rec = recall_score(y_test,y_pred, average='macro')
f1 = f1_score(y_test,y_pred, average='macro')
cm = confusion_matrix(y_pred,y_test)
report = classification_report(y_pred,y_test)

print('accuracy:' , acc)
print('precision:',pr)
print('recall:',rec)
print('f1 score :',f1)
print(cm)
print(report)
          
          
          
          
      
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
      
          
         
          
