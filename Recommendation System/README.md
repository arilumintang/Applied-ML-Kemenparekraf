# Laporan Proyek _Machine Learning_ - Dimas Ari Lumintang
## Latar Belakang

Sistem rekomendasi film (_movies recommendation_) adalah meningkatnya minat dan konsumsi konten film di era digital. Dengan adanya platform streaming dan jumlah film yang terus meningkat, pengguna seringkali menghadapi kesulitan dalam memilih film yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi film hadir sebagai solusi untuk membantu pengguna menemukan film-film yang relevan dan sesuai dengan minat mereka.

Sistem rekomendasi film bekerja dengan memanfaatkan data pengguna, seperti riwayat penontonan, preferensi film sebelumnya, atau rating yang diberikan. Algoritma rekomendasi kemudian menganalisis data tersebut untuk mengidentifikasi pola dan tren yang dapat digunakan untuk membuat rekomendasi yang personal dan akurat.

Dalam mengembangkan sistem rekomendasi film, terdapat beberapa pendekatan yang dapat digunakan, seperti _collaborative filtering_ (filtrasi kolaboratif) yang mengandalkan informasi dari pengguna lain yang memiliki minat serupa, atau _content-based filtering_ (filtrasi berbasis konten) yang menggunakan informasi tentang film itu sendiri, seperti genre, sutradara, atau aktor.

Dengan adanya sistem rekomendasi film, pengguna dapat dengan mudah menemukan film-film baru yang mungkin mereka sukai, menghemat waktu dalam mencari film yang sesuai, dan secara keseluruhan meningkatkan pengalaman menonton mereka. Sistem rekomendasi ini juga memberikan manfaat bagi penyedia platform streaming dan industri film secara keseluruhan dengan meningkatkan interaksi pengguna dan promosi konten yang lebih efektif.

# Business Understanding
### Problem Statement
   * Bagaimana cara mendapatkan rekomendasi film yang terbaik berdasarkan data dari penonton film?

### Goals
   * Membuat Sistem Rekomendasi untuk kumpulan film.

### Solution Approach
Solusi yang saya ajukan yaitu menggunakan algoritma machine learning untuk sistem rekomendasi yaitu:

* _Content Based Filtering_ adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit. Algoritma ini memberikan rekomendasi berdasarkan aktivitas pada masa lalu.
* _Collaborative Filtering_ adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Algoritma ini memberikan rekomendasi berdasarkan nilai rating atau nilai lain, disini saya menggunakan target sebagai dasar penilaian.

# Data Understanding
File-file ini berisi metadata untuk semua 45.000 film yang terdaftar di Kumpulan Data MovieLens Lengkap. Kumpulan data terdiri dari film yang dirilis pada atau sebelum Juli 2017. Poin data mencakup pemeran, kru, kata kunci plot, anggaran, pendapatan, poster, tanggal rilis, bahasa, perusahaan produksi, negara, penghitungan suara TMDB, dan rata-rata suara.

Kumpulan data ini juga memiliki file yang berisi 26 juta peringkat dari 270.000 pengguna untuk semua 45.000 film. Rating berada pada skala 1-5 dan telah didapatkan dari situs resmi GroupLens. [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

### Overview Data:
* movies_metadata.csv
  File Metadata Film utama. Berisi informasi tentang 45.000 film yang ditampilkan dalam kumpulan data Full MovieLens. Fitur termasuk poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, negara produksi, dan perusahaan.
* ratings_small.csv
  Subkumpulan 100.000 peringkat dari 700 pengguna di 9.000 film.

    
### Variabel-variabel pada _The Movies Dataset_ adalah sebagai berikut:
Dataset "The Movies Dataset" di Kaggle memiliki beberapa kolom yang mengandung informasi tentang film-film yang terdaftar dalam dataset tersebut. Berikut adalah penjelasan singkat tentang beberapa kolom utama dalam dataset tersebut:

* adult: Kolom ini menunjukkan apakah film tersebut ditujukan untuk penonton dewasa (True) atau tidak (False).
* budget: Kolom ini berisi informasi tentang anggaran produksi film dalam bentuk nilai numerik.
* genres: Kolom ini menyimpan informasi tentang genre-genre film. Setiap genre dipisahkan oleh karakter "|".
* id: Kolom ini merupakan identifikasi unik untuk setiap film dalam dataset.
* original_language: Kolom ini menunjukkan bahasa asli film tersebut.
* original_title: Kolom ini berisi judul asli film dalam bahasa aslinya.
* overview: Kolom ini berisi ringkasan atau deskripsi singkat tentang isi film.
* popularity: Kolom ini menunjukkan popularitas film dalam dataset, dihitung berdasarkan faktor-faktor seperti jumlah penonton, rating, dan aktivitas online.
* production_companies: Kolom ini menyimpan informasi tentang perusahaan produksi yang terlibat dalam pembuatan film.
* production_countries: Kolom ini berisi informasi tentang negara-negara tempat film diproduksi.
* release_date: Kolom ini menunjukkan tanggal rilis film.
* revenue: Kolom ini berisi informasi tentang pendapatan yang dihasilkan oleh film tersebut.
* runtime: Kolom ini menunjukkan durasi film dalam menit.
* spoken_languages: Kolom ini berisi informasi tentang bahasa-bahasa yang digunakan dalam film.
* title: Kolom ini berisi judul film dalam bahasa Inggris.
* vote_average: Kolom ini menunjukkan rata-rata rating yang diberikan oleh penonton terhadap film.
* vote_count: Kolom ini berisi jumlah total suara yang diterima oleh film.

Kemudian persebaran data statistik dataset ini adalah sebagai berikut:
Tabel 2. Statistical Description
| Describe | revenue       | runtime      | vote\_average | vote_count  |
| -------- | ------------- | ------------ | ------------ | ----------- |
| count    | 4.546000e+04  | 45203.000000 | 45460.000000 | 45460.000000|
| mean     | 1.120935e+07  | 94.128199    | 5.618207     | 109.897338  |
| std      | 6.433225e+07  | 38.407810    | 1.924216     | 491.310374  |
| min      | 0.000000e+00  | 0.000000     | 0.000000     | 0.000000    |
| 25%      | 0.000000e+00  | 85.000000    | 5.000000     | 3.000000    |
| 50%      | 0.000000e+00  | 95.000000    | 6.000000     | 10.000000   |
| 75%      | 0.000000e+00  | 107.000000   | 6.800000     | 34.000000   |
| max      | 2.787965e+09  | 1256.000000  | 10.000000    | 14075.000000|
Dari tabel di atas dapat dilihat bahwa hanya terdapat 4 data bertipe numerik, yaitu revenue, runtime, vote\_average, dan vote_count. Dengan tabel ini dapat diketahui hasil perhitungan statistik seperti mean, nilai minimum, nilai maksimum, dan lain-lain. Dengan menggunakan tabel ini dapat mempermudah melihat hal-hal seperti rentang nilai dari masing-masing data dan lain-lain.

### Visualization
Gambar 1 adalah persebaran histogram dari _The Movies Dataset_, fungsi grafik histogram digunakan untuk memvisualisasikan distribusi frekuensi dari suatu data. Distribusi frekuensi dari suatu data merujuk pada cara data tersebut terdistribusi atau tersebar dalam kelompok nilai-nilai yang berbeda. Distribusi frekuensi mencerminkan jumlah kemunculan atau frekuensi relatif dari setiap nilai dalam data tersebut. Histogram terdiri dari serangkaian batang vertikal, di mana sumbu x mewakili nilai-nilai data dan sumbu y mewakili frekuensi kemunculan nilai-nilai tersebut.
![histsr](https://github.com/arilumintang/Applied-ML-Kemenparekraf/assets/68593835/b1c55e40-65e3-466c-84bd-d65fd3f3cec2)
Gambar 1. Histogram distribution
Berdasarkan hasil histogram, terdapat beberapa variabel yang memiliki korelasi antara satu sama lain. Berikut adalah penjelasan korelasi dari masing-masing variabel:

1. Korelasi antara "revenue" (pendapatan) dan variabel lainnya:
    - "revenue" memiliki korelasi dengan "runtime" (durasi film) karena pendapatan film dapat dipengaruhi oleh lamanya film ditayangkan.
    - "revenue" juga memiliki korelasi dengan "vote_count" (jumlah suara) karena popularitas dan kesuksesan film dapat tercermin dari jumlah suara yang diberikan oleh penonton.
2. Korelasi antara "runtime" (durasi film) dan variabel lainnya:
    - "runtime" mungkin memiliki korelasi dengan "vote_average" (nilai rata-rata suara) karena durasi film yang ideal dapat mempengaruhi penilaian penonton terhadap film tersebut.
    - "runtime" juga dapat memiliki korelasi dengan "vote_count" karena film dengan durasi yang lebih lama cenderung menarik lebih banyak penonton, yang dapat mempengaruhi jumlah suara yang diberikan.
3. Korelasi antara "vote_average" (nilai rata-rata suara) dan "vote_count" (jumlah suara):
    - "vote_average" dan "vote_count" mungkin memiliki korelasi karena popularitas film yang tinggi cenderung mendapatkan jumlah suara yang lebih banyak, dan nilai rata-rata suara dapat tercermin dari jumlah suara yang diberikan.


Gambar 2 adalah hasil _correlation matrix_ dari _The Movies Dataset_ untuk memahami korelasi dari masing-masing variabel yang dimiliki.
![corrsr](https://github.com/arilumintang/Applied-ML-Kemenparekraf/assets/68593835/fa7b80a2-152d-43d7-a907-0e19ce0f95e4)
Gambar 2. Correlation matrix
Dari correlation matrix di atas, didapatkan pengetahuan bahwa:
* Nilai 1 menunjukkan korelasi sempurna secara positif, artinya ada hubungan langsung dan linier antara variabel.
* Nilai 0.26 dan nilai yang jauh dari 1 atau -1 menunjukkan korelasi positif atau negatif lemah antara variabel. Meskipun ada hubungan, namun tidak terlalu kuat atau linier.
* Nilai 0.82 dan nilai yang dekat dari 1 atau -1 menunjukkan korelasi positif atau negatif kuat antara variabel. Sebenarnya jika ada nilai korelasi yang mendekati 1 atau -1 ini, maka salah satu variabel yang berkorelasi kuat ini dapat dihapus, namun karena jumlah variabel yang sedikit, maka pengahpusan variabel tidak dilakukan karena dapat mengakibatkan pelatihan data menjadi tidak baik.
* Korelasi positif menunjukkan bahwa perubahan pada satu variabel cenderung berhubungan dengan perubahan searah pada variabel lainnya, sedangkan korelasi negatif menunjukkan hubungan berlawanan arah antara variabel.

# Data Preparation
Teknik _data preparation_ yang dilakukan pada penelitian ini adalah sebagai berikut:
## Missing value
Pada _dataset_ ini data numerik yang _missing_ diisi nilainya dengan nilai _mean_. Oleh karena itu, tidak ada data yang dikurangi (_drop_).

## Cek korelasi 
Pada awalnya akan dilakukan _data reduction_ dengan melihat nilai korelasi pada _correlation matrix_, namun karena variabel terlalu sedikit sehingga proses pengurangan data diurungkan. Hal ini bertujuan untuk mengurangi kemungkinan overfitting karena _data train_ terlalu sedikit. Variabel yang digunakan untuk dilihat korelasinya adalah revenue, runtime, vote\_average, vote\_count. Hanya 4 variabel tersebut karena hanya 4 variabel itu lah yang merupakan variabel dengan tipe data numerik.

## Seperate data
Hal ini dilakukan agar mempermudah ketika memanggil variabel-variabel bertipe data kategorik dan data numerik.
* Numeric column:
[revenue, runtime, vote\_average, vote\_count]
* Categoric column:
[adult, belongs\_to\_collection, budget, genres, homepage, id, imdb\_id, original\_language, original\_title, overview, popularity, poster\_path, production\_companies, production\_countries, release\_date, spoken\_languages, status, tagline, title, video]

# Modeling
Pada tahap ini, akan dikembangkan model _Machine Learning Content Based Filtering_ dan _Collaborative Filtering_ untuk memberikan rekomendasi musik.

## Content-Based Filtering
Metode Content-Based Filtering adalah salah satu pendekatan dalam sistem rekomendasi yang menggunakan karakteristik atau konten dari item atau objek yang akan direkomendasikan. Dalam konteks film, metode ini mencoba untuk merekomendasikan film-film yang memiliki kesamaan dalam konten atau fitur-fitur tertentu, seperti genre, aktor/aktris, sutradara, atau tema.

Salah satu teknik yang sering digunakan dalam metode Content-Based Filtering adalah cosine similarity (similaritas kosinus). Cosine similarity digunakan untuk mengukur sejauh mana kedekatan atau kesamaan antara dua vektor dalam ruang fitur. Dalam hal ini, setiap film direpresentasikan sebagai vektor fitur yang menggambarkan karakteristik atau atribut-atributnya. [2] Berikut rumus cosine similarity:
\sum_{i=1}^n * A_i B_i \over \sum_{i=1}^n * A_i^2 * \sum_{i=1}^n * B_i^2

* Data yang digunakan pada metode ini adalah data yang disukai oleh pengguna pada masa lalu. Rekomendasi yang dihasilkan merupakan rekomendasi yang berdasarkan data pengguna tersebut di masa lalu.

* Hasil top N Recommendation terhadap film berjudul "The Dark Knight Rises".

Tabel 2. Hasil rekomendasi
| ID   | title                    | Genre         |
| ---- | ------------------------ | ------------- |
| 31143 | Deadly Daycare          | ..., Drama    |
| 19286 | The One Percent         | ..., Drama    |
| 44918 | Once More               | ..., Drama    |
| 45106 | Nicostratos the Pelican | ..., Drama    |
| 33008 | White Cannibal Queen    | ..., Drama    |
| 15225 | Father and Son          | ..., Drama    |
| 44912 | Three Songs About Lenin | ..., Drama    |
| 4132  | The Luzhin Defence      | ..., Drama    |
| 34321 | Maine Pyar Kiya         | ..., Drama    |
| 9844  | Come and Get It         | ..., Drama    |

Di sini genre yang sama dari keseluruhan film rekomendasi yang muncul adalah genre drama.

## Collaborative Filtering
Metode Collaborative Filtering adalah sebuah pendekatan dalam sistem rekomendasi yang mengumpulkan informasi preferensi dari sekelompok pengguna untuk menghasilkan rekomendasi bagi pengguna yang serupa. Metode ini berfokus pada pola hubungan antara pengguna dan item yang dinilai oleh pengguna tersebut.

KNN (K-Nearest Neighbors) adalah salah satu algoritma yang digunakan dalam Collaborative Filtering. Algoritma ini bekerja dengan cara mencari K tetangga terdekat dari pengguna yang sedang direkomendasikan atau item yang sedang dijadikan acuan. KNN menggunakan metrik jarak (misalnya, Euclidean distance) untuk mengukur seberapa mirip antara pengguna atau item yang sedang dibandingkan. Berdasarkan jarak tersebut, KNN mengambil K tetangga terdekat dan menggunakan preferensi atau rating yang diberikan oleh tetangga tersebut untuk membuat rekomendasi.

* Data yang digunakan pada metode ini adalah data yang berupa nilai, biasanya rating. Disini saya meenggunakan kolom target / kolom yang merupakan pemicu bahwa user memutar lagunya kembali setelah 1 bulan sejak diputar pertama kali.

* Top N Recommendation yang dihasilkan sebagai berikut.
```
['Spy Kids', 'Stalker', 'Stake Land', 'Stagecoach', 'St. Vincent', 'Spy Kids: All the Time in the World', 'Stand by Me', 'Spy Kids 2: The Island of Lost Dreams', 'Spring, Summer, Fall, Winter... and Spring', 'Spring Breakers']
```

* Kelebihan Content-Based Filtering:
Sistem rekomendasi berbasis konten dapat menjelaskan bagaimana hasil rekomendasi didapatkan dan sistem rekomendasi berbasis konten dapat merekomendasikan itemitem yang bahkan belum pernah di-rate oleh siapapun. [3]

* Kekurangan Content-Based Filtering:
Kelemahan dari metode content-based filtering adalah terbatasnya rekomendasi hanya pada item-item yang mirip sehingga tidak ada kesempatan untuk mendapatkan item
yang tidak terduga. [2]

* Kelebihan Collaborative Filtering:
Metode collaborative filtering memiliki beberapa kelebihan yaitu rekomendasi tetap akan berkerja dalam keadaan dimana konten sulit dianalisi sekalipun. [3]

* Kekurangan Collaborative Filtering:
Metode collaborative filtering memiliki kekurangan yaitu membutuhkan parameter rating, sehingga jika ada item baru sistem tidak akan merekomendasikan item tersebut. [3]

# Evaluation
## Precision
Precision mengacu pada proporsi kasus positif yang diidentifikasi dengan benar dari keseluruhan hasil yang diberikan oleh sistem atau model. Metrik ini digunakan untuk mengukur sejauh mana sistem atau model dapat menghasilkan hasil yang akurat dan relevan dalam mengklasifikasikan kasus positif. Berikut rumus dari precision:

Precision = True Positives / (True Positives + False Positives)
Di mana:
* True positives (TP) adalah jumlah kasus positif yang diidentifikasi dengan benar oleh sistem.
* False positives (FP) adalah jumlah kasus negatif yang salah diidentifikasi sebagai positif oleh sistem.
* Hasil precision pada content based filtering penelitian ini adalah sebesar 100% 
Ada alasan yang mungkin menyebabkan nilai precision sebesar 100%, yaitu:
    - Jumlah rekomendasi yang sangat terbatas: Jika jumlah rekomendasi yang dihasilkan sangat sedikit dan semuanya tepat sesuai dengan preferensi pengguna, maka bisa mendapatkan nilai precision sebesar 100%. 
    - Data pengguna yang sangat spesifik: Jika sistem rekomendasi hanya berfokus pada satu pengguna dengan preferensi yang sangat spesifik, maka dapat mencapai nilai precision 100% untuk pengguna tersebut.

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

## Hasil Evaluasi
* Hasil evaluasi RMSE dan MSE rating_small.csv

| Metrik          | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean   | Std    |
|-----------------|--------|--------|--------|--------|--------|--------|--------|
| RMSE (testset)  | 0.9021 | 0.8986 | 0.8867 | 0.8995 | 0.8969 | 0.8968 | 0.0053 |
| MAE (testset)   | 0.6929 | 0.6941 | 0.6831 | 0.6929 | 0.6909 | 0.6908 | 0.0040 |

* Hasil evaluasi precision

Untuk Content Base Filtering saya saya akan menghitung precision nya dengan rumus berikut:
**recommender system precision = p $\text {of recommendations that are relevants} \over \text{of items we recommended}$**
Untuk cara menghitung nya disini saya meminta rekomendasi film dengan referensi "The Dark Knight Rises." Bisa dilihat di tabel 1 ada 10 dari 10 rekomendasi diberikan yang sesuai dalam kesamaan genre, artinya kita hitung precision nya dengan cara.
p = 10/10
p = 100%
Jadi, dapat dipastikan bahwa model content based filtering yang digunakan di sini benar-benar tepat dalam memberikan rekomendasinya.

## Kesimpulan
Setelah melalui serangkaian tahapan, disimpulkan bahwa kedua model yang penulis gunakan berhasil dalam memprediksi dengan akurasi yang memuaskan meskipun data yang digunakan terbatas. Tantangan utama yang dihadapi penulis adalah keterbatasan daya komputasi dalam mengembangkan sistem rekomendasi ini. Dari hasil evaluasi pada sistem rekomendasi film ini, tingkat presisi dari model sebesar 100%, yang mana artinya model ini dapat memberikan rekomendasi yang tepat kepada pengguna sesuai dengan film yang diinginkan.

# Daftar Referensi
[1] Dong Liu, J., W. Choi, dan J. Liu. 2021. Metode Rekomendasi Film yang Dipersonalisasi Berdasarkan Deep Learning. [https://www.hindawi.com/journals/mpe/2021/6694237/](https://www.hindawi.com/journals/mpe/2021/6694237/) [accessed June 10 2023]
[2] Resha Havilah, M., A. Wijayanto, dan Winarno. 2019. Recommendation System with Content-Based Filtering Method for Culinary Tourism in Mangan Application. _Jurnal Ilmiah Teknologi dan Informasi_. 8(2): 65-72.
[3] W. Andreas Eko dan D. Alfian. 2018. Sistem Rekomendasi Laptop Menggunakan
Collaborative Filtering dan Content-Based Filtering. _Jurnal Computech & Bisnis_. 12(1): 11-27