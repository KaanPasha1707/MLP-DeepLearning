# AU-AIR Veri Seti ile İHA Görüntülerinde MLP ve PyTorch Kullanarak Nesne Sınıflandırma

Bu proje, AU-AIR veri setindeki insansız hava aracı (İHA/Drone) görüntüleri üzerinde yer alan nesneleri, önceden eğitilmiş (pre-trained) özellik çıkarıcı (feature extractor) ağlar **kullanmadan**, doğrudan ham pikseller üzerinden Çok Katmanlı Algılayıcı (MLP - Multilayer Perceptron) mimarisi ile sınıflandırmayı amaçlamaktadır. Proje bütünüyle PyTorch kütüphanesi kullanılarak geliştirilmiş, hiperparametre optimizasyonları ve regülasyon teknikleri ile modelin genelleme yeteneği artırılmıştır.

**AU-AIR Veri Seti Görselleri:** https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCXJ

**AU-AIR Veri Seti Etiketleri:** https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA

## 🏗️ Proje Mimarisi ve İş Akışı

Sistem temel olarak dört aşamadan oluşmaktadır:

1. **Veri Ön İşleme:** `annotations.json` dosyasındaki koordinatlar kullanılarak her bir nesne resimden tek tek kırpılmıştır. Kırpılan resimler 48x48 piksel boyutuna yeniden ölçeklendirilmiş ve 6912 boyutlu (48 x 48 x 3) ham piksel vektörleri halinde düzleştirilerek numpy dizilerine (`.npy`) dönüştürülmüştür.
2. **PyTorch Veri Boru Hattı (Pipeline):** Özel bir `Dataset` sınıfı (`AUAIRDataset`) oluşturularak veriler PyTorch tensorlerine çevrilmiş ve `DataLoader` aracılığıyla modele paketler (batch) halinde beslenmiştir.
3. **Sınıf Dengesizliği Çözümü:** Veri setindeki aşırı dengesizliği önlemek için sınıfların frekansları sayılmış ve "ters karekök" formülüyle ağırlıklandırılarak doğrudan PyTorch'un `CrossEntropyLoss` fonksiyonuna entegre edilmiştir. Bu sayede azınlık sınıflarına verilen ceza katsayısı artırılarak ağın bu sınıflara daha fazla odaklanması sağlanmıştır.
4. **Model Mimarisi:** PyTorch ile esnek ve dinamik bir `DinamikMlp` sınıfı kodlanmıştır.

## 🧠 Dinamik MLP Tasarımı ve Optimizasyon

Donanım kısıtları altında gradyan patlamalarını önlemek ve en optimum performansa ulaşmak adına seçilen **nihai model tasarımı** şu şekildedir:

*   **Girdi Katmanı:** 6912 Nöron (Düzleştirilmiş ham piksel girdisi)
*   **Gizli Katman (H1):** 512 Nöron + Batch Normalization + ReLU + Dropout (0.4)
*   **Gizli Katman (H2):** 256 Nöron + Batch Normalization + ReLU + Dropout (0.4)
*   **Çıkış Katmanı:** 8 Nöron (Sınıf olasılık skorları)

**Optimizasyon Stratejileri:**
*   **Dropout:** İlk testlerdeki aşırı öğrenmeyi (overfitting) engellemek adına ağdaki Dropout oranı %40'a (0.4) çıkarılmış ve ağın belirli piksel gruplarına bağımlı kalması engellenmiştir.
*   **Erken Durdurma (Early Stopping):** 100 epoch olarak ayarlanan eğitim sürecinde, `valLoss` takip edilmiş ve sabır (patience) limiti 12 olarak belirlenmiştir.
*   **Öğrenme Oranı (Learning Rate):** Modelin hata uzayındaki lokal minimum noktalarını ıskalamadan kararlı adımlarla ilerlemesi için Adam optimizasyon algoritmasının öğrenme oranı 0.0005 seviyesine çekilmiştir.

## 📊 Veri Seti Dağılımı

Modelin aynı drone karesindeki nesneleri hem eğitim hem de test süreçlerinde görerek ezberlemesini önlemek amacıyla resim bazlı bölünme uygulanmıştır:
*   **Eğitim (Train):** 22.976 resim (nesne)
*   **Geçerleme (Validation):** 4.923 resim (nesne)
*   **Test:** 4.924 resim (nesne)

*(Not: Sınıflar; Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer'dan oluşmaktadır ve Car sınıfı %77.6 ile ezici bir çoğunluğa sahiptir.)*

## 🏆 Deneysel Sonuçlar

En iyi hiperparametre kombinasyonu (Konfigürasyon 4) eğitim tamamlandıktan sonra `EnİyiMlpModel.pt` olarak diske kaydedilmiş ve nihai test setine tabi tutulmuştur.

*   **Test Doğruluğu:** Sadece ham piksel verisi kullanılmasına ve herhangi bir CNN öznitelik çıkarıcı (ResNet, AlexNet vb.) kullanılmamasına rağmen model **%83.00** genel doğruluk (accuracy) elde etmiştir.
*   **Dengeli Öğrenme:** Ağırlıklandırılmış kayıp fonksiyonu sayesinde, ezici çoğunluğa sahip "car" sınıfının yanında, azınlık sınıfları olan "bus" (%60 F1-Skor) ve "motorbike" gibi sınıflarda da kayda değer başarımlar yakalanmıştır.
*   **Performans Kıyası:** Projenin önceki aşamalarındaki KNN (%89.00) ve Random Forest (%86.00) modellerinin hazır öznitelikler kullanmasına kıyasla; bu MLP modeli sadece ham piksellerle onlara oldukça yakın bir başarım sergilemiş, aynı zamanda ağırlık matrislerinden ibaret olduğu için operasyonel olarak çok daha yüksek bir çıkarım hızı (inference speed) sunmuştur.

## ⚙️ Kurulum ve Çalıştırma

**Gereksinimler:**
Projenin çalışması için `numpy`, `Pillow`, `scikit-learn` ve **`torch`** (PyTorch) kütüphanelerinin yüklü olması gerekmektedir.

1. Depoyu bilgisayarınıza klonlayın.
2. Gerekli bağımlılıkları sisteminize kurun.
3. `assignment3.py` dosyasını çalıştırarak öznitelik çıkarma, eğitim ve test sürecini başlatın.

> **⚠️ Önemli Not (Veri Boyutu):**
> Görüntü boyutlandırma sonucunda oluşan devasa boyutlu matris dosyaları (`.npy`) ve ağırlık modeli (`.pt`), GitHub'ın dosya sınırını aştığı için bu depoya yüklenmemiştir. `assignment3.py` dosyası çalıştırıldığında, resimlerden vektörleri çıkararak bu dosyaları bilgisayarınızda lokal olarak otomatik üretecek ve diskinize kaydedecektir.
> İndireceğiniz dosyaların isimleri koddakinden farklı olabilir. O isimlere göre kodu güncellemeyi unutmayın.
