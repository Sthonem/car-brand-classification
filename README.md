# car-brand-classification
CNN &amp; Transfer Learning ile Car Brand Classification – Akbank Deep Learning Bootcamp
🚗 Car Brand Classification with EfficientNet-B4

Bu proje, Akbank Derin Öğrenme Bootcamp kapsamında geliştirilmiştir.
Amaç, 33 farklı araç markasını CNN tabanlı bir derin öğrenme modeli ile sınıflandırmaktır.

⸻

📂 Veri Seti
	•	Dataset: Car Brand Classification Dataset (Kaggle)
	•	Train: 11,517 görsel
	•	Validation: 2,475 görsel
	•	Test: 2,475 görsel
	•	Toplam: 16,467 görsel
	•	Görseller 33 araç markasına aittir (Audi, BMW, Toyota, Mercedes, vs.)

⸻

🧠 Model Mimarisi
	•	Temel Model: EfficientNet-B4 (ImageNet pretrained)
	•	Katmanlar:
	•	Data Augmentation (Flip, Rotation, Zoom, Contrast, Translation)
	•	Preprocess (EfficientNet için normalize)
	•	EfficientNetB4 (include_top=False, global average pooling)
	•	Dropout (0.5)
	•	Dense Layer (33 sınıf için Softmax)

⸻

⚙️ Eğitim Stratejisi
	1.	Warm-up Training
	•	Base model donduruldu (frozen)
	•	Optimizer: Adam (lr=1e-4)
	•	Epochs: 15
	•	Callback: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
	2.	Fine-Tuning
	•	EfficientNetB4 son 200 katmanı açıldı (unfreeze)
	•	Optimizer: Adam (lr=1e-5)
	•	Epochs: 40
	•	Callback: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

⸻

📊 Sonuçlar
	•	Validation Accuracy (Top-1 / Top-5): 72.9% / 88.0%
	•	Test Accuracy (Top-1 / Top-5): 71.7% / 88.5%
	•	Macro F1-Score: ~0.71
	•	En iyi doğruluk: ~%72 (Top-1)

📈 Eğitim sürecinde overfitting’i engellemek için data augmentation ve dropout kullanılmıştır.

⸻

🖼️ Grad-CAM (Model Açıklanabilirliği)

Modelin karar verirken görselin hangi bölgelerine odaklandığı Grad-CAM tekniği ile görselleştirilmiştir:
	•	Aracın ön ızgara ve logo bölgesi genellikle yüksek aktivasyon göstermektedir.
	•	Bu, modelin gerçekten markayı ayırt edici görsel özelliklere odaklandığını göstermektedir.

⸻

🔮 Test Time Augmentation (TTA)

Test seti için TTA (flip, rotation, crop) uygulanarak doğruluk artırılmıştır.
	•	Normal test doğruluğu: 71.7%
	•	TTA sonrası doğruluk: ~73–74%

⸻

📊 Confusion Matrix
	•	Bazı markalar (örneğin MINI, FIAT, Chrysler) yüksek başarı ile tanınırken,
	•	Görsel olarak benzer markalarda (örneğin Ford vs. Toyota) daha fazla karışıklık görülmüştür.

⸻
🚀 Kullanım

Eğitim
python train.py

Modeli Yükleme
from tensorflow import keras
model = keras.models.load_model("effnetb4_carbrand_final.keras")

Tahmin
import tensorflow as tf
img = tf.keras.utils.load_img("test_image.jpg", target_size=(380,380))
x = tf.keras.utils.img_to_array(img)[None,...]
pred = model.predict(x)
print("Tahmin:", class_names[pred.argmax()])


kaggle linki: https://www.kaggle.com/code/sthonem/akbank3
