# Laporan Proyek _Machine Learning_ - Dimas Ari Lumintang
## Domain Proyek
Kredit adalah salah satu alat pembayaran yang umum digunakan dalam transaksi keuangan, baik secara fisik maupun online. Namun, dengan semakin meningkatnya penggunaan kartu kredit, juga muncul risiko penipuan dan kegiatan kejahatan terkait dengan kartu kredit, seperti _fraud_ atau penipuan kartu kredit.

Masalah _cedit card fraud_ menjadi signifikan karena dapat menyebabkan kerugian finansial yang besar bagi individu, perusahaan, dan bahkan sistem keuangan secara keseluruhan. Penipuan kartu kredit dapat melibatkan pencurian data kartu kredit, penggunaan kartu yang tidak sah, atau transaksi yang tidak diotorisasi oleh pemilik kartu.

Tujuan utama dalam menganalisis masalah _credit card fraud_ adalah mengembangkan model prediktif yang dapat mengidentifikasi transaksi yang mencurigakan atau _fraud_ secara efektif. Dengan menggunakan teknik-teknik analisis data dan pembelajaran mesin, kita dapat mempelajari pola dan perilaku transaksi yang normal serta membangun model yang dapat membedakan transaksi yang sah dari yang tidak sah.

Penerapan model prediktif untuk mendeteksi _credit card fraud_ dapat membantu lembaga keuangan, penyedia kartu kredit, dan pengguna kartu kredit untuk mengidentifikasi transaksi yang mencurigakan secara _real-time_. Dengan demikian, langkah-langkah keamanan tambahan dapat diambil untuk mencegah kerugian finansial yang lebih lanjut dan melindungi pengguna kartu kredit dari potensi penipuan. [1]

## Business Understanding
### Problem Statement
   * Bagaimana cara menentukan algoritma _machine learning_ yang baik dalam mengklasifikasikan permasalahan _credit card fraud_?
   * Bagaimana perbandingan hasil evaluasi metode _K-Nearest Neighbors_, _Random Forest_, dan _Boosting Algorithm_?

### Goals
   * Mengetahui algortima _machine learning_ yang baik dalam mengklasifikasian permasalahan _credit card fraud_.
   * Menganalisis perbandingan hasil evaluasi metode  _K-Nearest Neighbors_, _Random Forest_, dan _Boosting Algorithm_.
   * Untuk mengetahui algoritma _machine learning_ yang baik digunakan dalam klasifikasi _credit card fraud_ bisa dengan menerapkan beberapa metode klasifikasi.
   * Untuk menganalisa perbandingan hasil evaluasi metode _K-Nearest Neighbors_, _Random Forest_, dan _Boosting Algorithm_ dapat dilakukan dengan mengamati metrik evaluasi seperti:
     - _Train accuracy_
     - _Test accuracy_
     - _Confusion Matrix_
     - _Mean Squared Error_ (MSE)
     - _Root Mean Squared Error_ (RMSE)
     - _R-Squared_ (R2)

## Data Understanding
Data yang digunakan adalah data yang berasal dari kaggle, data ini berisikan Dataset berisi transaksi yang dilakukan oleh kartu kredit pada bulan September 2013 oleh pemegang kartu Eropa. [_Credit Card Fraud Detection_] (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Dataset ini menyajikan transaksi yang terjadi dalam dua hari, dimana  ditemukan 492 penipuan dari 284.807 transaksi. Dataset sangat tidak seimbang, kelas positif (penipuan) menyumbang 0,172% dari semua transaksi. 

_Dataset_ ini berisi variabel input numerik yang merupakan hasil dari transformasi PCA. Sayangnya, karena masalah kerahasiaan, _dataset_ ini tidak dapat memberikan fitur asli dan informasi latar belakang lainnya tentang data tersebut.

### Overview Data:
    - Dataset name: Credit Card Fraud Detection
    - Overall columns
        Valid: 285.000
        Mismatched: 0
        Missing: 0
    - Source: MACHINE LEARNING GROUP - ULB
    - License: Open Knowledge Foundation

### Variabel-variabel pada _Credit Card Fraud Detection dataset_ adalah sebagai berikut:
   * V1, V2, … V28 adalah komponen utama yang diperoleh dengan PCA.
   * Time adalah jumlah detik berlalu antara transaksi ini dan transaksi pertama dalam kumpulan data.
   * Amount adalah jumlah transaksi.
   * Class adalah variabel target yang terbagi menjadi 2, yaitu 1 untuk data _fraud_ dan 0 untuk _no fraud_.

### Analisis Predictive
Tabel 1. _Describe statistics_
| Parameter | Time | V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | ... | V28 | Amount | Class |
|-----------|------|----|----|----|----|----|----|----|----|----|-----|-----|--------|-------|
| count     | 284807 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | 2.848070e+05 | ... | 2.848070e+05 | 2.848070e+05 | 284807 |
| mean      | 94813.859575 | 1.168375e-15 | 3.416908e-16 | -1.379537e-15 | 2.074095e-15 | 9.604066e-16 | 1.487313e-15 | -5.556467e-16 | 1.213481e-16 | -2.406331e-15 | ... | -1.227390e-16 | 88.349619 | 0.001727 |
| std       | 47488.145955 | 1.958696e+00 | 1.651309e+00 | 1.516255e+00 | 1.415869e+00 | 1.380247e+00 | 1.332271e+00 | 1.237094e+00 | 1.194353e+00 | 1.098632e+00 | ... | 3.300833e-01 | 250.120109 | 0.041527 |
| min       | 0.000000 | -5.640751e+01 | -7.271573e+01 | -4.832559e+01 | -5.683171e+00 | -1.137433e+02 | -2.616051e+01 | -4.355724e+01 | -7.321672e+01 | -1.343407e+01 | ... | -1.543008e+01 | 0.000000 | 0.000000 |
| 25%       | 54201.500000 | -9.203734e-01 | -5.985499e-01 | -8.903648e-01 | -8.486401e-01 | -6.915971e-01 | -7.682956e-01 | -5.540759e-01 | -2.086297e-01 | -6.430976e-01 | ... | -5.295979e-02 | 5.600000 | 0.000000 |
| 50%       | 84692.000000 | 1.810880e-02 | 6.548556e-02 | 1.798463e-01 | -1.984653e-02 | -5.433583e-02 | -2.741871e-01 | 4.010308e-02 | 2.235804e-02 | -5.142873e-02 | ... | 1.124383e-02 | 22.000000 | 0.000000 |
| 75%       | 139320.500000 | 1.315642e+00 | 8.037239e-01 | 1.027196e+00 | 7.433413e-01 | 6.119264e-01 | 3.985649e-01 | 5.704361e-01 | 3.273459e-01 | 5.971390e-01 | ... | 7.827995e-02 | 77.165000 | 0.000000 |
| max       | 172792.000000 | 2.454930e+00 | 2.205773e+01 | 9.382558e+00 | 1.687534e+01 | 3.480167e+01 | 7.330163e+01 | 1.205895e+02 | 2.000721e+01 | 1.559499e+01 | ... | 3.384781e+01 | 25691.160000 | 1.000000 |
* Hasil analisis:
    * Data terdiri dari 8 baris dan 31 kolom.
    * Data terdiri dari 284.807.
    * Rata-rata jumlah transaksi yang terjadi adalah 88,35.
    * Jumlah transaksi terendah adalah sebanyak 0 dan tertinggi sebanyak 25.691,16.

### Visualization
Gambar 1 adalah persebaran histogram dari _dataset credit card fraud detection_ untuk memahami distribusi data, mendeteksi _outliers_, mendeteksi rentang nilai, dan lainnya.
![hist](https://github.com/arilumintang/ML-Kemenparekraf/assets/68593835/0c938ec1-2f21-4954-962c-06c8d57c9615)
Gambar 1. Histogram distribution

Gambar 2 adalah hasil _correlation matrix_ dari _dataset credit card fraud detection_ untuk memahami korelasi dari masing-masing variabel yang dimiliki.
![corrmat](https://github.com/arilumintang/ML-Kemenparekraf/assets/68593835/ec809c7e-04a5-454d-90c7-fb21456c6437)
Gambar 2. Correlation matrix

# Data Preparation
Teknik _data preparation_ yang dilakukan pada penelitian ini adalah sebagai berikut:
* Variabel target pada _dataset_ ini sudah dalam bentuk biner "1" untuk _fraud_ dan "0" untuk _no fraud_, sehingga proses OHE bisa dilewatkan.
* Nilai-nilai _dataset_ ini sudah diproses dengan _principal component analysis_ (PCA) sehingga proses data reduction dengan PCA bisa dilewatkan.
* Data reduction yang dilakukan dilihat melalui _correlation matrix_, nilai korelasi yang dikurangi adalah variabel dengan nilai korelasi di atas 0,5 dan di bawah -0,5. Dari _correlation matrix_ didapatkan variabel "V2" dan "Amount" memiliki korelasi -0,53, maka diputuskan bahwa variabel "V2" akan di-_drop_.
* StandardScaler, ini adalah metode normalisasi yang digunakan pada penelitian ini. Tujuan dari StandardScaler adalah untuk mengubah fitur-fitur tersebut sehingga memiliki rata-rata nol (mean = 0) dan simpangan baku satu (standard deviation = 1). StandardScaler ini membantu dalam menyeimbangkan skala variabel-variabel yang memiliki rentang nilai yang berbeda. Normalisasi data dapat membantu menghindari efek dominasi dari variabel yang memiliki skala yang lebih besar.
* TrainTestSplit, ini digunakan untuk membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model.
* SMOTE, salah satu teknik yang digunakan untuk menangani masalah ketidakseimbangan kelas dalam dataset. Teknik ini menghasilkan sampel sintetis untuk kelas minoritas dengan cara menggabungkan data dari kelas minoritas yang ada. Selain mengatasi ketidak seimbangan data, dengan menggunakan SMOTE model dapat meningkatkan representasi kelas minoritas, penggunaan SMOTE dapat membantu model mempelajari pola-pola yang lebih baik dari kelas minoritas. Ini dapat meningkatkan kinerja model dalam mengenali dan memprediksi kelas minoritas dengan lebih akurat.

Berikut adalah tabel final data setelah dilakukan normalisasi data dengan StandardScaler dan data reduction.
Tabel 2. _Describe statistics_ setelah normaliasi
| Parameters | Time      | V1        | V3        | V4        | V5        | V6        | V7        | V8        | V9        | ... | V28       | Amount    | Class        |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----|-----------|-----------|--------------|
| count      | 284807.0  | 284807.0  | 284807.0  | 284807.0  | 284807.0  | 284807.0  | 284807.0  | 284807.0  | 284807.0  | ... | 284807.0  | 284807.0  | 284807.0     |
| mean       | -3.065637e-16 | -1.506872e-17 | -8.622104e-17 | -5.189230e-18 | 3.832046e-17 | 9.979288e-18 | 1.237432e-17 | -3.193372e-18 | 7.234983e-19 | ... | 2.913952e-17 | 0.001727  |
| std        | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | 1.000002e+00 | ... | 1.000002e+00 | 0.041527  |
| min        | -1.996583e+00 | -2.879855e+01 | -3.187173e+01 | -4.013919e+00 | -8.240810e+01 | -1.963606e+01 | -3.520940e+01 | -6.130252e+01 | -1.222802e+01 | ... | -3.532294e-01 | 0.000000  |
| 25%        | -8.552120e-01 | -4.698918e-01 | -5.872142e-01 | -5.993788e-01 | -5.010686e-01 | -5.766822e-01 | -4.478860e-01 | -1.746805e-01 | -5.853631e-01 | ... | -3.308401e-01 | 0.000000  |
| 50%        | -2.131453e-01 | 9.245351e-03 | 1.186124e-01 | -1.401724e-02 | -3.936682e-02 | -2.058046e-01 | 3.241723e-02 | 1.871982e-02 | -4.681169e-02 | ... | -2.652715e-01 | 0.000000  |
| 75%        | 9.372174e-01 | 6.716939e-01 | 6.774569e-01 | 5.250082e-01 | 4.433465e-01 | 2.991625e-01 | 4.611107e-01 | 2.740785e-01 | 5.435305e-01 | ... | -4.471707e-02 | 0.000000  |
| max        | 1.642058e+00 | 1.253351e+00 | 6.187993e+00 | 1.191874e+01 | 2.521413e+01 | 5.502015e+01 | 9.747824e+01 | 1.675153e+01 | 1.419494e+01 | ... | 1.023622e+02 | 1.000000  |

Tabel 2 adalah hasil dari deskripsi data setelah _dataset_ diproses dengan StandardScaler dan _data reduction_. Efeknya adalah menyeimbangkan skala variabel-variabel yang memiliki rentang nilai yang berbeda dan pengurangan variabel "V2" dapat lebih mempercepat waktu komputasi.

Untuk _train test tplit_ kita bisa menggunakan potongan kode berikut:
```
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X = data.drop('Class', axis=1)
y = data['Class']
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

# Modeling
Pada tahap ini, akan dikembangkan model _Machine Learning_ dengan melakukan perbandingan 3 algoritma, yaitu _K-Nearest Neighbors_, _Random Forest_, dan _Boosting Algorithm_. 4 algoritma tersebut akan dievaluasi performa di tahap _evaluation_ untuk menentukan model terbaik.
## _Install models and metrics library_
Pertama _install_ library yang dibutuhkan untuk mempermudah proses modelin dan evaluasi, berikut library yang diinstall:
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
```
Tabel 3. Hasil model
| Models             | Train Accuracy | Test Accuracy |
| ------------------ | -------------- | ------------- |
| KNN                | 0.999          | 0.999         |
| Random Forest      | 1.000          | 0.999         |
| Gradient Boosting  | 0.994          | 0.994         |

Dari hasil pada Tabel 3 di atas, semua model mendapatkan hasil akurasi yang baik. Akan tetapi dapat dilihat bahwa algoritma _random forest_ memberikan hasil paling baik karena mendapatkan nilai _train accuracy_ 1,000 dan _test accuracy_ 0,999. Sedangkan algoritma yang paling jelek dari ketiga model yang digunakan adalah _gradient boosting_, dengan hasil _train accuracy_ 0,994 dan _test accuracy_ 0,994.
## Model yang digunakan
### Models
1. K-Nearest Neighbors
Algoritma KNN merupakan metode yang digunakan untuk melakukan  klasifikasi  data berdasarkan jarak terpendek terhadap objek data. Penentuan nilai  K yang  terbaik  untuk  algoritma   ini berdasarkan pada data yang ada. Nilai K yang tinggi dapat mengurangi efek noise pada  klasifikasi,  bisa juga membuat   batasan   antara setiap   klasifikasi   menjadi lebih kabur. [2]
    * Kelebihan:
        - Sederhana dan mudah diimplementasikan.
        - Tidak memerlukan asumsi tentang distribusi data.
        - Mampu menangani data dengan fitur yang kompleks atau non-linear.
        - Fleksibel dalam menangani masalah klasifikasi dan regresi.
        - Kinerja yang baik pada dataset dengan ukuran kecil hingga sedang.
    * Kekurangan:
        - Sensitif terhadap data pencilan (outliers) karena menghitung jarak terhadap tetangga terdekat.
        - Membutuhkan penghitungan jarak terhadap semua titik data saat melakukan prediksi, yang dapat menjadi komputasi yang mahal pada dataset yang besar.
        - Tidak efisien dalam hal ruang karena membutuhkan penyimpanan seluruh dataset pelatihan.
        - Perlu menentukan parameter k untuk jumlah tetangga yang akan digunakan, yang bisa mempengaruhi kinerja algoritma.
    * Parameter:
        - n_neighbors = 5
    * Algoritma KNN:
    ![algoritma-knn](https://github.com/arilumintang/ML-Kemenparekraf/assets/68593835/1ccd2db0-2b63-4291-9b18-ba11076c6493)
2. Random Forest
_Random forest_ adalah kombinasi pohon prediktor sedemikian rupa sehingga setiap pohon bergantung pada nilai sebuah sampel vektor acak secara independen dan dengan distribusi yang sama untuk semua pohon di _forest_. [3]
    * Kelebihan:
        - Tahan terhadap _overfitting_.
        - Stabilitas dan kinerja yang tinggi.
        - Kemampuan menangani data yang tidak seimbang.
        - Kemampuan untuk mengevaluasi pentingnya fitur.
    * Kekurangan:
        - Interpretabilitas yang terbatas.
        - Membutuhkan lebih banyak sumber daya.
        - Tidak efektif untuk data dengan dimensi tinggi.
    * Parameter:
        - n_estimator = 100
        - max_depth = Infinite / None (node diperluas sampai semua semua daun kurang dari sampel)
    * Algoritma Random Forest:
    ![Random_forest_diagram_complete](https://github.com/arilumintang/ML-Kemenparekraf/assets/68593835/6eb496a8-1eca-4953-a8b6-cd6bc422f9c1)
3. Boosting Algorithm
Gradient boosting merupakan algoritma klasifikasi machine learning yang menggunakan ensamble dari decision tree untuk memprediksi nilai. Gradient boosting termasuk supervised learning berbasis decision tree yang dapat digunakan untuk klasifikasi. Gradient boosting dimulai dengan menghasilkan pohon klasifikasi awal dan terus menyesuaikan pohon baru melalui minimalisasi fungsi kerugian. [4]
    * Kelebihan:
        - Menangani berbagai jenis data.
        - Menangani interaksi fitur.
        - Tidak terlalu sensitif terhadap preprocessing.
    * Kekurangan:
        - Waktu pelatihan yang lama.
        - Kemungkinan overfitting.
        - Sensitif terhadap parameter.
    * Parameter:
        - n_estimators = 100 (Default)
        - subsample = None (Default)
        - max_depth = None (Default)
    * Algoritma Gradient Boosting:
    ![Schematical-representation-of-gradient-boosting-regression-in-regards-to-algorithm](https://github.com/arilumintang/ML-Kemenparekraf/assets/68593835/b4567661-361b-47f3-8a72-abd56d7614ee)

# Evaluation
Metrik untuk evaluasi dan _confusion matrix_ akan digunakan sebagai alat evaluasi pada penelitian ini, berikut adalah list nya:
1. Mean Squared Error (MSE)
2. Root Mean Squared Error (RMSE)
3. R2
4. Confussion Matrix

## Mean Squared Error (MSE)
MSE (Mean Squared Error) adalah salah satu metrik evaluasi yang umum digunakan dalam masalah regresi. Ini mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya dalam data. Semakin rendah nilai MSE, semakin baik performa model.

Rumus MSE dapat dituliskan sebagai berikut:

MSE = (1 / n) * Σ(yi - ŷi)^2

di mana:
- MSE: Mean Squared Error
- n: jumlah sampel dalam data
- yi: nilai sebenarnya dari sampel ke-i
- ŷi: nilai prediksi dari sampel ke-i
- Σ: simbol sigma untuk menjumlahkan seluruh kuadrat selisih antara nilai sebenarnya dan prediksi

## Root Mean Squared Error (RMSE)
RMSE (Root Mean Squared Error) adalah metrik evaluasi yang umum digunakan dalam pemodelan regresi untuk mengukur sejauh mana selisih antara nilai prediksi dan nilai sebenarnya. RMSE menghitung akar kuadrat dari rata-rata kesalahan kuadrat antara nilai prediksi dan nilai sebenarnya.

Rumus RMSE dapat dituliskan sebagai berikut:

RMSE = sqrt(mean((y\_pred - y\_actual)^2))

di mana:
- n adalah jumlah sampel.
- Σ adalah tanda sigma yang menunjukkan penjumlahan.
- y_pred adalah nilai prediksi.
- y_actual adalah nilai sebenarnya.

## R-Squared (R2)
R2 (R-squared), juga dikenal sebagai koefisien determinasi, adalah ukuran statistik yang digunakan untuk mengukur seberapa baik model regresi cocok dengan data yang diamati. R2 menggambarkan proporsi variabilitas dalam variabel dependen yang dapat dijelaskan oleh variabel independen dalam model regresi. Nilai R2 berkisar antara 0 hingga 1, di mana nilai 0 menunjukkan bahwa model tidak dapat menjelaskan variabilitas sama sekali, dan nilai 1 menunjukkan bahwa model dapat menjelaskan seluruh variabilitas.

Rumus R2 dapat dituliskan sebagai berikut:

R2 = 1 - (SSR / SST)

di mana:
- SSR (Sum of Squared Residuals) adalah jumlah kuadrat selisih antara nilai prediksi model dan nilai aktual dari variabel dependen.
- SST (Total Sum of Squares) adalah jumlah kuadrat selisih antara setiap nilai aktual dan nilai rata-rata dari variabel dependen.

## Analisa Model
1. Hasil metrik

| Model             | MSE           | RMSE          | R2            |
| ----------------- | ------------- | ------------- | ------------- |
| KNN               | 0.0002286197  | 0.0151201741  | 0.9990855195  |
| Random Forest     | 0.0000527584  | 0.0072634967  | 0.9997889660  |
| Gradient Boosting | 0.0063134200  | 0.0794570323  | 0.9747462689  |

Berdasarkan nilai evaluasi yang diberikan, model Random Forest memiliki performa terbaik. Hal ini dapat dilihat dari MSE (Mean Squared Error) dan RMSE (Root Mean Squared Error) yang memiliki nilai yang lebih rendah, serta R2 Score yang memiliki nilai yang lebih tinggi dibandingkan dengan model K-Nearest Neighbors dan Gradient Boosting. R2 Score mendekati nilai 1 menunjukkan bahwa model Random Forest memberikan prediksi yang sangat baik terhadap data yang digunakan. Oleh karena itu, model Random Forest dapat dianggap sebagai yang terbaik di antara ketiga model tersebut dalam hal performa.
2. Hasil _confusion matrix_
* KNN

|               | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| Actual Negative | 56756 | 26 |
| Actual Positive | 0 | 56944 |
    
* Random Forest

|               | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| Actual Negative | 56776 | 6 |
| Actual Positive | 0 | 56944 |

* Gradient Boosting

|               | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| Actual Negative | 56281 | 501 |
| Actual Positive | 217 | 56727 |

Dari hasil _confusion matrix_ di atas dapat dilihat bahwa metode _random forest_ memberikan hasil terbaik karena dapat menentukan data yang benar-benar _fraud_ dan _no fraud_. Serta metode _random forest_ memberikan nilai _false positive_ dan _false negative_ paling sedikit.

Maka, dapat disimpulkan bahwa metode _Random Forest_ merupakan metode terbaik jika dibandingkan dengan metode _K-Nearest Neighbors_ dan _Gradient Boosting_.
# Daftar Referensi
[1] Jiayin, Z. 2022. Credit Card Fraud Detection Using Predictive Model. _Jurnal of Business and Management_. 38: 2820-2825.
[2] Cholil, S.R., T. Handayani, R. Prathivi, dan T. Ardianita. 2021. Implementasi Algoritma Klasifikasi K-Nearest Neighbor (KNN) Untuk Klasifikasi Seleksi Penerima Beasiswa. _Indonesian Journal on Computer and Information Technology_. 6(2):118-127.
[3] Breiman, L. 2022. Random Forest. _Machine Learning_. 45: 5-32.
[4] Febryananda, A.R. 2022. Mengenal Teknik Gradient Boosting dalam Algoritma Machine Learning. Laboratorium Analisis Data dan Rekaya Kualitas. [link](https://lab_adrk.ub.ac.id/id/mengenal-teknik-gradient-boosting-dalam-algoritma-machine-learning/). [accessed May. 29 2023]