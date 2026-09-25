import os, numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json, random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

with open("C:/Users/USER/OneDrive/Belgeler/auair2019annotations/annotations.json", "r") as file:
    tümEtiketler = json.load(file)

etiketHaritası = {}
for e in tümEtiketler['annotations']:
    resimAd = e['image_name'] 
    bbox = e.get('bbox',[])   

    if len(bbox) > 0:
        etiketHaritası[resimAd] = bbox

imageKlasör = "C:/Users/USER/OneDrive/Belgeler/auair2019data (1)/images"     
tümResimler = os.listdir(imageKlasör)

nesneliResimler = []
for n in tümResimler:
    if n in etiketHaritası:
        nesneliResimler.append(n)

print(f"toplam {len(tümResimler)} resimden {len(nesneliResimler)} tanesinde nesne bulundu")   

random.seed(3)
random.shuffle(nesneliResimler)

toplamResimSayısı = len(nesneliResimler)
trainDataSınır = int(toplamResimSayısı * 0.7)
validationDataSınır = int(toplamResimSayısı * 0.85)

trainResimleri = nesneliResimler[:trainDataSınır]
validationResimleri = nesneliResimler[trainDataSınır:validationDataSınır]
testResimleri = nesneliResimler[validationDataSınır:]

print(f"Resim dağılımı -> Train: {len(trainResimleri)}, Validation: {len(validationResimleri)}, Test: {len(testResimleri)}")

def vektörDöndürme(resimListesi, mod):
    vektörler = []
    etiketler = []
    count = 0

    for isim in resimListesi:
        yol = os.path.join(imageKlasör, isim)
        mevcutBboxlar = etiketHaritası.get(isim, [])

        try:
            tamResim = Image.open(yol).convert('RGB')
            for box in mevcutBboxlar:
                sınıf = box['class']
                top = int(box['top'])
                left =  int(box['left'])
                width = int(box['width'])
                height = int(box['height'])

                kırpılmışResim = tamResim.crop((left, top, left + width, top + height))
                boyutluResim = kırpılmışResim.resize((48,48))

                numpy = np.array(boyutluResim, dtype=np.float32) / 255.0
                vektör = numpy.flatten()

                vektörler.append(vektör)
                etiketler.append(sınıf)
            count += 1
            if count % 500 == 0:
               print(f"ilerleme: {count}")
        except Exception as e:
            continue

    np.save(f"mlp{mod}Vektör.npy", np.array(vektörler))
    np.save(f"mlp{mod}Etiket.npy", np.array(etiketler))    
    print(f"{mod} seti başarıyla kaydedildi. Vektör boyutu: {np.array(vektörler).shape}")

if not os.path.exists("mlpTrainVektör.npy"):
    print("Ham piksellerden 6912 boyutlu öznitelik vektörleri çıkarılıyor...")
    vektörDöndürme(trainResimleri, "Train")
    vektörDöndürme(validationResimleri, "Val")
    vektörDöndürme(testResimleri, "Test")

trainResim, trainEtiket = np.load("mlpTrainVektör.npy"), np.load("mlpTrainEtiket.npy")
valResim, valEtiket = np.load("mlpValVektör.npy"), np.load("mlpValEtiket.npy")
testResim, testEtiket = np.load("mlpTestVektör.npy"), np.load("mlpTestEtiket.npy")

sınıflar = sorted(list(set(trainEtiket)))
sınıfIndex = {}

for index, sınıf in enumerate(sınıflar):
    sınıfIndex[sınıf] = index

trainIndex = np.array([sınıfIndex[y] for y in trainEtiket])
valIndex = np.array([sınıfIndex[y] for y in valEtiket])
testIndex = np.array([sınıfIndex[y] for y in testEtiket])

sınıfSayıları = Counter(trainIndex)
toplam = len(trainIndex)

ağırlık = [np.sqrt(toplam/sınıfSayıları[i]) for i in range(len(sınıflar))]
ağırlıkTensor = torch.FloatTensor(ağırlık).to('cuda' if torch.cuda.is_available() else 'cpu')

class AUAIRDataset(Dataset):
    def __init__(self, girdi, etiket):
        self.girdi = girdi
        self.etiket = etiket

    def __len__(self):
        return len(self.etiket)

    def __getitem__(self, index):
        giren = torch.tensor(self.girdi[index], dtype=torch.float32)
        etiketParcası = torch.tensor(self.etiket[index], dtype=torch.long)
        return giren, etiketParcası

class DinamikMlp(nn.Module):
    def __init__(self, girdiBoyut, hiddenLayers, sınıflar, frekans, batch, dropout):
        super(DinamikMlp, self).__init__()
        layers = []
        boyut = girdiBoyut   
        
        for dim in hiddenLayers:
            layers.append(nn.Linear(boyut, dim))
            if batch:
                layers.append(nn.BatchNorm1d(dim))

            if frekans == 'relu':
                layers.append(nn.ReLU())
            else:
                if frekans == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                else:
                    if frekans == 'tanh':
                        layers.append(nn.Tanh())

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            boyut = dim                    
        
        layers.append(nn.Linear(boyut, sınıflar))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
    
def modelEgit(konfigürasyon, train, val, agırlık, dim = 6912, sınıflar = len(sınıflar)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DinamikMlp(
        girdiBoyut=dim,
        hiddenLayers=konfigürasyon['hiddenLayers'],
        sınıflar=sınıflar,
        frekans=konfigürasyon['frekans'],
        batch=konfigürasyon['batch'],
        dropout=konfigürasyon['dropout']
    ).to(device)    

    kriter = nn.CrossEntropyLoss(weight=agırlık)
    optimizer = optim.Adam(model.parameters(), lr=konfigürasyon['lr'])

    patience = konfigürasyon['patience']
    pcounter = 0
    enİyiValLoss = float('inf')
    enİyiModel = None
    enİyiAccuracy = 0.0

    for epoch in range(100):

        model.train()
        trainLoss = 0.0

        for batchx, batchy in train:
            batchx, batchy = batchx.to(device), batchy.to(device)

            optimizer.zero_grad()
            output = model(batchx)

            loss = kriter(output, batchy)
            loss.backward()
            optimizer.step()

            trainLoss = trainLoss + loss.item() * batchx.size(0)

        trainLoss = trainLoss/len(train.dataset)

        model.eval()
        valLoss = 0.0
        tahminler = []
        etiketler = []

        with torch.no_grad():
            for batchx, batchy in val:
                batchx, batchy = batchx.to(device), batchy.to(device)
                output = model(batchx)
                loss = kriter(output, batchy)
                valLoss = valLoss + loss.item() * batchx.size(0)

                deger, aitSınıf = torch.max(output, 1)
                tahminler.extend(aitSınıf.cpu().numpy())
                etiketler.extend(batchy.cpu().numpy())

        valLoss = valLoss/len(val.dataset)
        valAccuracy = accuracy_score(etiketler, tahminler)

        print(f"Epoch {epoch+1:02d} -> Train Loss: {trainLoss:.4f} , Validation Loss: {valLoss:.4f} , Validation Accuracy: {valAccuracy:.4f}")

        if valLoss < enİyiValLoss:
            enİyiValLoss = valLoss
            enİyiAccuracy = valAccuracy
            enİyiModel = model.state_dict()
            pcounter = 0
        else:
            pcounter +=1
            if pcounter >= patience:
                print(f"Early stopping tetiklendi! Epoch {epoch+1} de eğitim durduruldu")
                break                

    return enİyiModel, enİyiAccuracy

trainDataset = AUAIRDataset(trainResim, trainIndex)
valDataset = AUAIRDataset(valResim, valIndex)
testDataset = AUAIRDataset(testResim, testIndex)

hiperparametreler = [

    #hiperparametre optimizasyonu öncesi
    #{'batchSize': 64, 'lr': 0.001, 'patience': 7, 'hiddenLayers': [512,256], 'frekans': 'relu', 'batch': True, 'dropout': 0.2},
    #{'batchSize': 64, 'lr': 0.0001, 'patience': 7, 'hiddenLayers': [512,256], 'frekans': 'leaky_relu', 'batch': True, 'dropout': 0.3},
    #{'batchSize': 128, 'lr': 0.001, 'patience': 10, 'hiddenLayers': [1024,512,256], 'frekans': 'relu', 'batch': True, 'dropout': 0.5},

    #hiperparametre optimizasyonu sonrası
    {'batchSize': 64, 'lr': 0.0005, 'patience': 12, 'hiddenLayers': [512,256], 'frekans': 'relu', 'batch': True, 'dropout': 0.4},
    {'batchSize': 64, 'lr': 0.0003, 'patience': 10, 'hiddenLayers': [256,128], 'frekans': 'leaky_relu', 'batch': True, 'dropout': 0.3},
    {'batchSize': 128, 'lr': 0.0001, 'patience': 12, 'hiddenLayers': [512,128], 'frekans': 'tanh', 'batch': True, 'dropout': 0.2}
]

enİyiSkor = -1
enİyiKonfigürasyon = None
enİyiAgırlık = None

print("--- MLP Hiperparametre Optimizasyon Süreci Başlıyor ---")
for index, konfigürasyon in enumerate(hiperparametreler):
    print(f"\nKonfigürasyon {index+1} eğitiliyor: {konfigürasyon}")

    trainLoad = DataLoader(trainDataset, batch_size=konfigürasyon['batchSize'], shuffle=True)
    valLoad = DataLoader(valDataset, batch_size=konfigürasyon['batchSize'], shuffle=False)

    model, valAccuracy = modelEgit(konfigürasyon,trainLoad,valLoad,ağırlıkTensor)
    print(f"Validation Doğruluğu: {valAccuracy:.4f}")

    if valAccuracy > enİyiSkor:
        enİyiSkor = valAccuracy
        enİyiKonfigürasyon = konfigürasyon
        enİyiModel = model

print("--- Optimizasyon Tamamlandı: En İyi Model Test Ediliyor ---")
print(f"En İyi Konfigürasyon: {enİyiKonfigürasyon}")

torch.save(enİyiModel, "EnİyiMlpModel.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
finalModel = DinamikMlp(
    girdiBoyut=6912,
    hiddenLayers=enİyiKonfigürasyon['hiddenLayers'],
    sınıflar=len(sınıflar),
    frekans=enİyiKonfigürasyon['frekans'],
    batch=enİyiKonfigürasyon['batch'],
    dropout=enİyiKonfigürasyon['dropout']
).to(device)

finalModel.load_state_dict(torch.load("EnİyiMlpModel.pt"))
finalModel.eval()

testLoad = DataLoader(testDataset, batch_size=enİyiKonfigürasyon['batchSize'], shuffle=False)
tahminler = []
testEtiketler = []
    
with torch.no_grad():
    for batchx, batchy in testLoad:
        batchx = batchx.to(device)
        output = finalModel(batchx)
        deger, aitSınıf = torch.max(output,1)
        tahminler.extend(aitSınıf.cpu().numpy())
        testEtiketler.extend(batchy.cpu().numpy())

orijinalİsimler = ['human','car','truck','van','motorbike','bicycle','bus','trailer']

print("\n --- MLP Modelinin Test Seti Sınıflandırma Raporu ---")
print(classification_report(testEtiketler, tahminler, target_names=orijinalİsimler))

print("--- MLP Confusion Matrix ---")
print(confusion_matrix(testEtiketler, tahminler))
