# Laporan Proyek Machine Learning - Nuansa Syafrie Rahardian

## 1. Domain Proyek

Penyakit kardiovaskular merupakan penyebab utama kematian di seluruh dunia, dengan gagal jantung sebagai salah satu komplikasi paling serius. Gagal jantung terjadi ketika jantung tidak dapat memompa darah secara efektif untuk memenuhi kebutuhan tubuh. Menurut World Health Organization (WHO), sekitar 17,9 juta orang meninggal setiap tahunnya akibat penyakit kardiovaskular, dan jumlah ini terus meningkat seiring bertambahnya usia populasi dan perubahan gaya hidup.

Diagnosis dini sangat penting untuk mencegah progresivitas gagal jantung. Namun, tantangan utama adalah keterbatasan sumber daya medis, terutama di daerah terpencil. Tidak semua fasilitas kesehatan memiliki alat diagnostik yang canggih atau tenaga ahli yang memadai. Akibatnya, banyak pasien baru diketahui mengalami gagal jantung saat kondisinya sudah parah.

Untuk mengatasi hal tersebut, teknologi kecerdasan buatan, khususnya machine learning, menjadi solusi potensial. Dengan menganalisis data klinis sederhana seperti tekanan darah, denyut jantung, kadar kolesterol, dan riwayat penyakit, algoritma machine learning dapat digunakan untuk memprediksi risiko gagal jantung sejak dini. Sistem prediktif ini tidak menggantikan dokter, melainkan menjadi alat bantu pengambilan keputusan dalam proses skrining awal.

Proyek ini bertujuan membangun model machine learning untuk memprediksi gagal jantung berdasarkan data historis pasien. Dengan pendekatan berbasis data, diharapkan dapat mendukung tenaga medis dalam meningkatkan kecepatan dan akurasi diagnosis, serta efisiensi pelayanan kesehatan.

## 2. Business Understanding

Tantangan utama dalam penanganan gagal jantung adalah keterlambatan diagnosis akibat keterbatasan alat dan tenaga medis. Kebutuhan akan sistem prediktif yang cepat, akurat, dan mudah diimplementasikan menjadi sangat penting, terutama untuk fasilitas kesehatan dengan sumber daya terbatas.

Proyek ini bertujuan untuk:

- Mengembangkan model machine learning yang dapat memprediksi risiko gagal jantung berdasarkan data klinis pasien.

- Memberikan sistem pendukung keputusan medis untuk meningkatkan deteksi dini.

- Membantu rumah sakit dalam memprioritaskan penanganan pasien berisiko tinggi.

Dengan model prediktif ini, diharapkan dapat meningkatkan efisiensi operasional dan menurunkan angka kematian akibat gagal jantung melalui intervensi dini.




### Problem Statements

Dalam dunia medis, khususnya di bidang kardiologi, deteksi dini terhadap potensi penyakit jantung merupakan langkah krusial yang dapat menyelamatkan nyawa. Penyakit jantung masih menjadi penyebab utama kematian di banyak negara, dan sering kali terlambat terdiagnosis karena keterbatasan alat atau tenaga medis di fasilitas kesehatan tingkat pertama.

Dataset yang digunakan dalam proyek ini mencakup berbagai parameter medis pasien seperti usia, jenis kelamin, tekanan darah, kolesterol, detak jantung maksimum, serta hasil tes elektrokardiogram dan uji stres. Dengan data ini, kita berupaya menjawab beberapa pertanyaan utama sebagai dasar pengembangan solusi prediktif:

- **Bagaimana membangun model klasifikasi biner yang mampu memprediksi keberadaan penyakit jantung berdasarkan parameter medis yang tersedia?**  
  Hal ini mencakup proses mulai dari eksplorasi data, pembersihan, transformasi, hingga pelatihan model machine learning dan evaluasi performanya.

- **Faktor medis mana saja yang paling signifikan terhadap diagnosis penyakit jantung?**  
  Pemahaman terhadap fitur yang paling berpengaruh, seperti jenis nyeri dada (ChestPainType), kadar kolesterol, tekanan darah, atau bentuk kemiringan segmen ST (ST_Slope), sangat penting untuk meningkatkan interpretabilitas model dan memberi nilai tambah bagi tenaga medis.

- **Model machine learning apa yang paling efektif untuk klasifikasi penyakit jantung dalam kasus ini?**  
  Karena ini adalah masalah klasifikasi biner (penyakit jantung: ya/tidak), maka perlu dibandingkan berbagai algoritma seperti, Random Forest, SVM, dan XGBoost guna memilih yang paling akurat dan efisien.

Ketiga pertanyaan ini menjadi dasar utama dalam perancangan pipeline machine learning serta validasi hasilnya secara kuantitatif dan interpretatif.

---

### Goals

Tujuan dari proyek ini terbagi dalam dua kategori utama: **tujuan praktis** dan **tujuan teknis**, yang saling mendukung dalam membangun sistem prediksi penyakit jantung berbasis data.

#### Tujuan Praktis

- Menyediakan sistem prediktif berbasis machine learning yang dapat membantu tenaga medis dalam mendeteksi kemungkinan penyakit jantung secara lebih cepat dan efisien.
- Memberikan alat bantu skrining awal terhadap pasien berdasarkan data medis dasar yang dapat diperoleh secara non-invasif.
- Mendukung pengambilan keputusan medis yang lebih baik, terutama di wilayah yang minim akses terhadap alat diagnostik lanjutan seperti EKG digital atau echocardiography.

#### Tujuan Teknis

- Mengembangkan model klasifikasi biner yang dapat memprediksi status penyakit jantung dengan menggunakan fitur-fitur seperti `Age`, `Sex`, `ChestPainType`, `Cholesterol`, `MaxHR`, dan `ST_Slope`.
- Melakukan preprocessing menyeluruh terhadap data, termasuk encoding data kategorikal, normalisasi fitur numerik, dan handling outliers jika diperlukan.
- Mengevaluasi performa berbagai algoritma machine learning menggunakan metrik seperti **accuracy**, **precision**, **recall**, **F1-score**, dan **ROC-AUC**.
- Melakukan feature importance analysis untuk mengidentifikasi variabel yang paling berkontribusi terhadap prediksi.
- Menyediakan visualisasi data dan hasil model yang intuitif, mudah dipahami oleh non-teknisi (seperti dokter umum), serta dapat dijadikan referensi klinis awal.

Melalui tujuan ini, proyek diharapkan tidak hanya menjadi eksperimen teknis semata, tetapi juga memberikan dampak nyata dalam konteks pelayanan kesehatan preventif.

---

### Solution Statements

Untuk menjawab permasalahan dan mencapai tujuan yang telah ditetapkan, pendekatan solusi akan mencakup tahapan-tahapan berikut:

1. **Exploratory Data Analysis (EDA)**  
   Menganalisis distribusi nilai, hubungan antar fitur, serta identifikasi fitur yang dominan terhadap variabel target `HeartDisease`.

2. **Preprocessing Data**  
   Meliputi transformasi fitur kategorikal menjadi numerik (melalui one-hot encoding atau label encoding), normalisasi fitur numerik, dan penanganan nilai ekstrem (outliers) jika diperlukan.

3. **Pemilihan dan Pelatihan Model**  
   Beberapa algoritma klasifikasi akan diuji, seperti:
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost  
   Setiap model akan diuji menggunakan data latih dan divalidasi dengan teknik seperti **cross-validation**.

4. **Evaluasi dan Interpretasi Model**  
   Metrik evaluasi akan digunakan untuk menilai kinerja model. Selain itu, interpretasi fitur (feature importance) akan dilakukan untuk memahami peran tiap parameter medis terhadap prediksi.

5. **Visualisasi Hasil**  
   Visualisasi seperti confusion matrix, ROC curve, dan plot korelasi antar fitur akan digunakan untuk mendukung interpretasi hasil dan komunikasi kepada pihak non-teknis.

Dengan pendekatan ini, diharapkan dapat dibangun sistem prediksi yang akurat, transparan, dan aplikatif di lingkungan dunia nyata, khususnya pada praktik medis berbasis data.


## 3. Data Understanding

Dataset yang digunakan dalam proyek ini berjudul "Heart Failure Prediction Dataset " dan diperoleh dari platform Kaggle, merupakan hasil penggabungan dari lima dataset penyakit jantung yang berasal dari sumber terpercaya, yakni:

- Cleveland Clinic Foundation (303 observasi)
- Hungarian Institute of Cardiology (294 observasi)
- University Hospital, Zurich (123 observasi)
- V.A. Medical Center, Long Beach (200 observasi)
- Stalog (Heart) Data Set (270 observasi)

Dataset dapat diakses melalui tautan berikut: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

Langkah awal dalam membangun model prediksi adalah memahami struktur dan karakteristik data yang digunakan. Dataset berisi 918 entri dengan 12 kolom fitur yang menggambarkan berbagai parameter medis pasien, baik numerik maupun kategorikal, serta satu kolom target yaitu `HeartDisease`.

Sample data yang terdapat pada dataset adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/bb1dd2ea-ffd9-422f-8281-906d63a69636)

Infromasi dataset tersebut dapat dilihat pada gambar dibawah ini:

![image](https://github.com/user-attachments/assets/6b282bdc-1dde-48f7-bdfe-421d5c8cfdf0)

Dari gambar yang ditampilkan, terdapat 1 variabel bertipe float64, 6 variabel bertipe int64, dan 5 variabel bertipe object (kategorikal)

### Deskripsi Variabel
Dataset ini memiliki 12 variabel/fitur/atribut dengan tipe data dan deskripsi sebagai berikut:

| No | Nama Atribut      | Tipe Data | Deskripsi                                                                                                   |
|----|-------------------|-----------|--------------------------------------------------------------------------------------------------------------|
| 1  | Age               | Integer   | Usia pasien dalam satuan tahun.                                                                              |
| 2  | Sex               | Object    | Jenis kelamin pasien.                                                                                        |
| 3  | ChestPainType     | Object    | Tipe nyeri dada yang dialami oleh pasien.                                                                    |
| 4  | RestingBP         | Integer   | Tekanan darah pasien saat dalam kondisi istirahat, diukur dalam mm Hg.                                       |
| 5  | Cholesterol       | Integer   | Kadar kolesterol total dalam darah pasien, diukur dalam mg/dL.                                               |
| 6  | FastingBS         | Integer   | Indikator kadar gula darah pasien saat puasa.                                                                |
| 7  | RestingECG        | Object    | Hasil pemeriksaan elektrokardiogram (EKG) saat pasien beristirahat.                                          |
| 8  | MaxHR             | Integer   | Detak jantung maksimum yang dicapai pasien selama aktivitas fisik atau uji stres.                            |
| 9  | ExerciseAngina    | Object    | Menunjukkan apakah pasien mengalami angina (nyeri dada) saat melakukan aktivitas fisik.                      |
| 10 | Oldpeak           | Float     | Ukuran depresi segmen ST setelah latihan fisik, digunakan sebagai indikator iskemia miokard.                 |
| 11 | ST_Slope          | Object    | Kemiringan segmen ST selama uji stres, digunakan untuk menilai kondisi jantung pasien.                       |
| 12 | HeartDisease      | Integer   | Label target yang menunjukkan ada atau tidaknya penyakit jantung pada pasien.                                |


Semua fitur dalam dataset ini relevan secara medis dan memungkinkan untuk dijadikan input dalam membangun model prediksi gagal jantung.
Melalui pemahaman awal terhadap data ini, kita dapat merancang proses analisis yang lebih terarah, memilih teknik preprocessing yang sesuai, serta membangun model prediktif yang mampu menghasilkan performa optimal.

---

#### Relevansi Dataset dalam Konteks Prediksi Penyakit Jantung

Dataset *Heart Failure Prediction* ini sangat relevan untuk analisis prediktif di bidang kesehatan kardiovaskular karena mencakup berbagai atribut penting yang berkaitan langsung dengan kondisi jantung seseorang. Beberapa fitur utama seperti hasil pemeriksaan elektrokardiogram (RestingECG), kadar kolesterol, tekanan darah, serta riwayat nyeri dada (ChestPainType), memberikan informasi klinis yang krusial dalam proses diagnosis penyakit jantung. Selain itu, atribut seperti `Oldpeak`, `MaxHR`, dan `ExerciseAngina` menggambarkan respons fisiologis tubuh terhadap aktivitas fisik, yang sering dijadikan indikator keberadaan penyakit arteri koroner atau disfungsi jantung.

Dataset ini juga mencakup variabel gaya hidup dan riwayat kesehatan yang memengaruhi risiko penyakit jantung, seperti kebiasaan berolahraga (yang tercermin dari MaxHR dan ExerciseAngina), serta status metabolik seperti diabetes (FastingBS). Oleh karena itu, data ini sangat cocok untuk digunakan dalam studi korelasi, eksplorasi hubungan antar faktor risiko, serta pengembangan model prediktif berbasis machine learning.

Secara keseluruhan, struktur dan cakupan atribut pada dataset ini mendukung berbagai pendekatan analisis mulai dari klasifikasi biner (apakah seseorang memiliki penyakit jantung atau tidak), hingga interpretasi fitur penting yang dapat meningkatkan transparansi model dan kepercayaan pengguna dalam konteks aplikasi klinis. Kombinasi antara data klinis, gaya hidup, dan indikator fisiologis menjadikan dataset ini sangat berharga dalam pengembangan sistem deteksi dini berbasis kecerdasan buatan untuk meningkatkan kualitas pelayanan kesehatan kardiovaskular.

--- 


### Pengecekan Missing Value dan Duplicate Data (Duplikasi Data)

Pada tahap ini kita akan mengecek data yang tidak valid pada dataset. Setelah diperiksa apakah terdapat kolom yang bernilai null, hasilnya adalah tidak ada. Sedangkan data duplikat atau data ganda juga tidak ada. Maka dengan demikian data siapa untuk dianalisis pada tahap selanjutnya.

---

### Pengecekan Outlier pada Kolom Numerik
Pengecekan outlier dilakukan pada kolom numerik untuk mengidentifikasi nilai-nilai ekstrem yang berpotensi memengaruhi hasil analisis dan model prediktif. Metode yang digunakan dalam pengecekan outlier adalah **Interquartile Range (IQR)**, di mana perhitungan IQR dilakukan dengan rumus sebagai berikut:

\[
IQR = Q3 - Q1
\]

dengan:
- **Q1 (Quartile 1)**: Nilai tengah dari data di bawah median (25% data).
- **Q3 (Quartile 3)**: Nilai tengah dari data di atas median (75% data).

Batas untuk mendeteksi outlier ditentukan oleh:
- **Batas Bawah** = Q1 - 1.5 × IQR
- **Batas Atas** = Q3 + 1.5 × IQR

Data yang berada di bawah atau di atas batas tersebut dianggap sebagai *outlier*. Dari hasil pengecekan, terdapat beberapa data yang termasuk outlier seperti yang ditampilkan pada gambar berikut:

![image](https://github.com/user-attachments/assets/c5c1e613-fcfc-4014-b043-933db6c82877)

---



## Exploratory Data Analysis (EDA)

### Univariate Analysis EDA

Ada beberapa tahap yang akan kita lakukan, yakni:
Tahap pertama, membagi variabel-variabel menjadi 2 jenis, yaitu variabel numerikal dan variabel kategorikal. Berikut merupakan kolom-kolom yang termasuk dalam variabel numerikal maupun kategorikal. <br>

![image](https://github.com/user-attachments/assets/8acac3b6-06db-4467-876a-ece1c0c78d93)

![image](https://github.com/user-attachments/assets/04e3dc00-2a70-44f3-9bcf-ce73887349f0)

### Tabel Atribut Kategorikal
Berikut ini adalah informasi detail terkait fitur-fitur kategorikal yang terdapat pada dataset **Heart Failure Prediction Dataset**:

| **No** | **Kolom Kategorikal** | **Jumlah Kategori Unik** | **Daftar Kategori**                                           |
|--------|------------------------|---------------------------|---------------------------------------------------------------|
| 1      | Sex                    | 2                         | M, F                                                         |
| 2      | ChestPainType          | 4                         | TA, ATA, NAP, ASY                                            |
| 3      | RestingECG             | 3                         | Normal, ST, LVH                                              |
| 4      | ExerciseAngina         | 2                         | Y, N                                                         |
| 5      | ST_Slope               | 3                         | Up, Flat, Down                                               |

### Penjelasan Kategori pada Fitur Kategorikal

Fitur-fitur kategorikal pada dataset ini memiliki peran yang signifikan dalam proses diagnosis dan prediksi penyakit jantung. Berikut adalah deskripsi masing-masing fitur serta makna dari kategori-kategorinya:

#### 1. `Sex`
- **Kategori:** `M` (Male), `F` (Female)
- **Penjelasan:** Jenis kelamin penting karena prevalensi dan gejala penyakit jantung dapat berbeda antara laki-laki dan perempuan. Misalnya, laki-laki memiliki risiko yang lebih tinggi mengalami serangan jantung pada usia lebih muda dibanding perempuan.

#### 2. `ChestPainType`
- **Kategori:** 
  - `TA`: Typical Angina  
  - `ATA`: Atypical Angina  
  - `NAP`: Non-Anginal Pain  
  - `ASY`: Asymptomatic
- **Penjelasan:** Ini merupakan salah satu indikator klinis utama. Nyeri dada (angina) adalah gejala klasik penyakit jantung koroner.
  - `Typical Angina (TA)` menggambarkan nyeri dada khas yang timbul saat aktivitas fisik dan mereda saat istirahat.
  - `Atypical Angina (ATA)` memiliki karakteristik yang tidak khas namun masih berhubungan dengan stres jantung.
  - `Non-Anginal Pain (NAP)` menandakan nyeri dada yang tidak berkaitan langsung dengan jantung.
  - `Asymptomatic (ASY)` berarti pasien tidak merasakan nyeri dada sama sekali, meskipun mungkin memiliki gangguan jantung—ini sering terjadi pada penderita diabetes atau lansia.

#### 3. `RestingECG`
- **Kategori:** `Normal`, `ST`, `LVH`
- **Penjelasan:** Menunjukkan hasil elektrokardiogram (ECG) saat istirahat.
  - `Normal`: Tidak ada kelainan deteksi dari ECG.
  - `ST`: Adanya kelainan gelombang ST-T yang bisa menjadi tanda iskemia atau infark miokard.
  - `LVH`: Menunjukkan adanya pembesaran (hipertrofi) ventrikel kiri yang berpotensi meningkatkan risiko gagal jantung dan aritmia.

#### 4. `ExerciseAngina`
- **Kategori:** `Y` (Yes), `N` (No)
- **Penjelasan:** Menunjukkan apakah pasien mengalami angina (nyeri dada) selama uji stres fisik.
  - `Yes`: Nyeri muncul saat latihan fisik, menandakan keterbatasan suplai darah ke jantung.
  - `No`: Tidak muncul gejala angina saat berolahraga, mengindikasikan fungsi jantung yang relatif normal dalam kondisi stres.

#### 5. `ST_Slope`
- **Kategori:** `Up`, `Flat`, `Down`
- **Penjelasan:** Bentuk kemiringan segmen ST selama uji stres, digunakan untuk menilai sejauh mana jantung merespon stres.
  - `Up`: Menandakan peningkatan segmen ST saat latihan, biasanya dianggap normal.
  - `Flat`: Tidak ada perubahan kemiringan, yang dapat menjadi tanda adanya penyakit jantung.
  - `Down`: Penurunan segmen ST, yang seringkali mengindikasikan iskemia miokard (kekurangan suplai darah ke jantung).

### Statistik Deskriptif Kolom Numerik

Untuk mendapatkan pemahaman awal mengenai karakteristik data numerik, tahap selanjutnya dilakukan analisis statistik deskriptif terhadap seluruh fitur numerik dalam dataset. Langkah ini menghasilkan ringkasan statistik yang mencakup nilai minimum, maksimum, rata-rata (mean), standar deviasi (std), serta kuartil (25%, 50%, dan 75%) untuk setiap kolom numerik. Statistik ini memberikan gambaran umum tentang sebaran data, potensi keberadaan nilai ekstrem (outlier), serta kecenderungan pusat data (tendensi sentral). Misalnya, nilai mean dan median yang berbeda secara signifikan dapat mengindikasikan sebaran data yang tidak simetris atau adanya skewness. Informasi ini menjadi landasan penting sebelum masuk ke tahap eksplorasi lanjutan dan preprocessing, karena memungkinkan peneliti untuk mengidentifikasi fitur-fitur yang mungkin memerlukan transformasi, normalisasi, atau penanganan khusus lainnya.

Hasil analisa deksriptif dapat dilihat pada gambar berikut:
![image](https://github.com/user-attachments/assets/809ff4ed-3211-4319-9aba-4e7e3050f5a2)

#### 🔹 **Age (Usia)**

- **Rata-rata usia pasien** adalah sekitar **53,5 tahun**, dengan rentang usia dari **28 hingga 77 tahun**.
- Sebagian besar pasien berusia antara **47 (Q1)** dan **60 tahun (Q3)**, menunjukkan bahwa mayoritas data berfokus pada kelompok usia dewasa hingga lanjut usia.
- Karena usia merupakan faktor risiko utama penyakit jantung, distribusi usia ini cukup representatif untuk studi prediksi penyakit jantung.


#### 🔹 **RestingBP (Tekanan Darah Saat Istirahat)**

- Nilai tekanan darah bervariasi dari **0 hingga 200 mm Hg**, dengan rata-rata sekitar **132 mm Hg**.
- Nilai minimum **0** secara medis tidak logis dan kemungkinan besar merupakan data yang hilang atau error input, sehingga perlu ditangani lebih lanjut.
- Kuartil menunjukkan bahwa sebagian besar pasien memiliki tekanan darah antara **120 hingga 140 mm Hg**, yang merupakan rentang borderline hingga hipertensi ringan.

#### 🔹 **Cholesterol (Kolesterol)**

- Kadar kolesterol berkisar antara **0 hingga 603 mg/dL**, dengan rata-rata sekitar **198,8 mg/dL**.
- Nilai **0** pada kolesterol juga tidak realistis secara klinis dan perlu ditinjau sebagai potensi error atau missing value.
- Rentang interkuartil (IQR) antara **173 hingga 267 mg/dL** menunjukkan bahwa sebagian besar pasien memiliki kadar kolesterol dalam ambang risiko sedang hingga tinggi.

#### 🔹 **FastingBS (Gula Darah Puasa > 120 mg/dL)**

- Merupakan fitur biner (0 atau 1) dengan nilai rata-rata **0,23**, menunjukkan bahwa sekitar **23%** pasien memiliki kadar gula darah puasa tinggi.
- Diabetes merupakan komorbiditas penting dalam kasus penyakit jantung, sehingga fitur ini relevan dalam analisis risiko.


#### 🔹 **MaxHR (Detak Jantung Maksimum)**

- Detak jantung maksimum berkisar dari **60 hingga 202 bpm**, dengan rata-rata sekitar **136 bpm**.
- Nilai ini sangat bervariasi tergantung usia dan kebugaran masing-masing pasien.
- Rentang IQR antara **120 hingga 156 bpm** menunjukkan bahwa sebagian besar pasien memiliki detak jantung maksimum dalam kisaran fisiologis normal saat menjalani uji stres.


#### 🔹 **Oldpeak (Depresi Segmen ST)**

- Nilai `oldpeak` bervariasi dari **-2.6 hingga 6.2**, dengan rata-rata **0.89**.
- Nilai negatif menandakan **peningkatan segmen ST**, sedangkan nilai positif menunjukkan **depresi segmen ST** yang bisa mengindikasikan iskemia miokard.
- Sebagian besar pasien memiliki nilai `oldpeak` antara **0.0 hingga 1.5**, menandakan gangguan ringan hingga sedang.


#### 🔹 **HeartDisease (Label/Target)**

- Nilai rata-rata label adalah **0.553**, berarti sekitar **55,3%** pasien dalam dataset ini dikategorikan memiliki penyakit jantung.
- Dataset tergolong cukup **seimbang** antara kelas positif dan negatif, sehingga cocok untuk pendekatan **klasifikasi biner** tanpa perlu penyesuaian besar seperti oversampling atau undersampling.


Tahap ketiga, Pada tahap ini, kita akan membuat visualisasi data kategorikal dalam bentuk grafik dengan menggunakan library python matplotlib dan seaborn. Hasilnya seperti gambar dibawah ini:

![download (28)](https://github.com/user-attachments/assets/d20b13b4-6d88-4f26-a0f1-4af2d8038c52)


Interpretasi:
| **Kolom**          | **Interpretasi**                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Sex**            | Mayoritas responden adalah laki-laki (`M`), menunjukkan ketimpangan distribusi gender.                          |
| **ChestPainType**  | Tipe nyeri dada yang paling umum adalah `ASY` (Asymptomatic), sedangkan `TA` (Typical Angina) paling sedikit.   |
| **RestingECG**     | Sebagian besar hasil EKG saat istirahat adalah `Normal`, diikuti oleh `LVH` dan `ST`.                           |
| **ExerciseAngina** | Lebih banyak individu yang tidak mengalami angina saat berolahraga (`N`) dibanding yang mengalami (`Y`).        |
| **ST\_Slope**      | Distribusi kemiringan segmen ST menunjukkan bahwa `Flat` dan `Up` mendominasi, sedangkan `Down` relatif jarang. |



Tahap keempat, kita akan membuat visualisasi data numerikal dalam bentuk grafik dengan menggunakan library python `matplotlib`. Hasilnya seperti gambar dibawah ini:

![download (27)](https://github.com/user-attachments/assets/00c43335-59d7-46ca-8623-a1f4ea808cb9)

Interprestasi:
| **Kolom**        | **Interpretasi**                                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Age**          | Distribusi usia mendekati normal, dengan puncak antara 50–60 tahun.                                                                                                      |
| **RestingBP**    | Sebagian besar nilai tekanan darah saat istirahat berada pada kisaran 120–140 mmHg, dengan beberapa nilai ekstrem 0 yang perlu diperiksa (kemungkinan data tidak valid). |
| **Cholesterol**  | Sebaran kolesterol cenderung positif skewed, dengan banyak nilai tinggi (di atas 400 mg/dL), menunjukkan kemungkinan outlier.                                            |
| **FastingBS**    | Sebagian besar nilai adalah `0`, menandakan mayoritas responden tidak memiliki kadar gula darah puasa > 120 mg/dL.                                                       |
| **MaxHR**        | Detak jantung maksimum menunjukkan distribusi normal, berkisar antara 100–160 bpm.                                                                                       |
| **Oldpeak**      | Mayoritas nilai berada di sekitar 0–2, namun terdapat nilai ekstrim hingga 6.2, menandakan potensi outlier.                                                              |
| **HeartDisease** | Distribusi target hampir seimbang, sedikit lebih banyak responden yang memiliki penyakit jantung (`1`).                                                                  |


### Multivariate Analysis EDA


Dalam eksplorasi data ini, kita melakukan **Multivariate Exploratory Data Analysis (EDA)**, yaitu proses menganalisis lebih dari dua variabel secara bersamaan untuk memahami hubungan antar fitur dan bagaimana mereka mempengaruhi target variabel, yaitu **HeartDisease**. Multivariate EDA sangat berguna dalam proses awal pemodelan karena dapat memberikan insight mengenai fitur mana yang paling relevan.

---

#### 1. Heatmap Korelasi Antar Fitur Numerik

![download (29)](https://github.com/user-attachments/assets/70680371-3e57-40fe-a699-da6f55f9773f)

Visualisasi ini menampilkan **korelasi Pearson** antar fitur numerik. Warna merah menunjukkan korelasi positif, sedangkan biru menunjukkan korelasi negatif.

- `HeartDisease` memiliki korelasi positif cukup kuat dengan `Oldpeak` (0.40) dan korelasi negatif dengan `MaxHR` (-0.40).
- Korelasi `Age` terhadap `HeartDisease` juga positif (0.28), yang menunjukkan semakin tua, kecenderungan memiliki penyakit jantung meningkat.
- Hubungan antar fitur saling tumpang tindih, menegaskan perlunya pendekatan multivariate dalam analisis.

---

#### 2. Pairplot Fitur Numerik

![download (32)](https://github.com/user-attachments/assets/6f07ab4a-9d5b-43ef-b5a3-0c140cbfcfd0)

Pairplot ini memvisualisasikan hubungan antar semua fitur numerik dan target `HeartDisease`. Setiap scatterplot menunjukkan relasi dua fitur, dengan distribusi marginal di sepanjang diagonal.

- Terdapat distribusi yang berbeda antara pasien dengan dan tanpa penyakit jantung, khususnya pada `MaxHR` dan `Oldpeak`.
- Dapat dilihat bahwa kombinasi fitur tertentu membentuk pola yang bisa dimanfaatkan dalam klasifikasi.

---

#### 3. Boxplot Fitur Numerik terhadap HeartDisease
![download (31)](https://github.com/user-attachments/assets/9faddd09-0cf9-4aba-9465-9f8c5feebda4)


Boxplot ini menunjukkan distribusi fitur numerik terhadap kelas `HeartDisease`.

- Pasien dengan penyakit jantung cenderung memiliki `Oldpeak` yang lebih tinggi dan `MaxHR` yang lebih rendah.
- Perbedaan signifikan juga terlihat pada distribusi `Age`, `Cholesterol`, dan `RestingBP` antara pasien dengan dan tanpa penyakit jantung.

---

#### 4. Korelasi Cramér's V untuk Fitur Kategorikal

![download (30)](https://github.com/user-attachments/assets/b475dba7-b5c5-440b-8074-5e96f1ec4d26)

Cramér's V digunakan untuk mengukur kekuatan asosiasi antara fitur kategorikal.

- `ExerciseAngina` dan `ST_Slope` memiliki asosiasi yang relatif kuat (0.46), serta `ChestPainType` juga memiliki korelasi dengan keduanya.
- Korelasi antar fitur kategorikal perlu dipertimbangkan agar tidak terjadi multikolinearitas saat modeling.

---
#### 5. Visualisasi Hubungan Fitur Kategorikal dengan HeartDisease (Cramér's V)

![download (33)](https://github.com/user-attachments/assets/82e664f1-912c-4079-9385-0e8a1a1dec0c)

Gambar di bawah ini menunjukkan **nilai Cramér’s V** antara setiap fitur kategorikal dan target variabel `HeartDisease`.  
Nilai Cramér’s V mengukur kekuatan asosiasi antara dua variabel kategorikal, dengan rentang dari **0 (tidak ada asosiasi)** hingga **1 (asosiasi sempurna)**.

Tabel Nilai Cramér’s V

| Fitur Kategorikal | Cramér’s V | Interpretasi           |
|-------------------|------------|-------------------------|
| **ST_Slope**      | 0.62       | Hubungan kuat           |
| **ChestPainType** | 0.54       | Hubungan cukup kuat     |
| **ExerciseAngina**| 0.49       | Hubungan sedang         |
| **Sex**           | 0.30       | Hubungan lemah-moderat  |
| **FastingBS**     | 0.26       | Hubungan lemah          |
| **RestingECG**    | 0.11       | Hampir tidak ada hubungan |

 Insight

- **ST_Slope** memiliki asosiasi paling kuat dengan penyakit jantung (`HeartDisease`). Ini menunjukkan bahwa bentuk perubahan segmen ST pada ECG sangat berkorelasi dengan keberadaan penyakit jantung.
- **ChestPainType** dan **ExerciseAngina** juga menunjukkan hubungan yang cukup kuat, yang selaras dengan pemahaman klinis bahwa jenis nyeri dada dan respons terhadap olahraga merupakan indikator penting.
- **Sex** dan **FastingBS** menunjukkan korelasi yang lebih lemah, meskipun masih ada hubungan.
- **RestingECG** tampaknya tidak memiliki korelasi signifikan terhadap variabel target.

#### Kesimpulan Akhir - Multivariate EDA Penyakit Jantung

Berdasarkan eksplorasi data multivariat yang telah dilakukan, berikut adalah beberapa temuan utama:

-  **Fitur Numerik yang Paling Berpengaruh** terhadap `HeartDisease` adalah:
  - `Oldpeak` – menunjukkan tingkat depresi segmen ST, sangat berkorelasi dengan adanya penyakit jantung.
  - `MaxHR` – detak jantung maksimum selama uji latihan, berpengaruh signifikan.
  - `Age` – risiko penyakit jantung meningkat dengan bertambahnya usia.
  - `FastingBS` – kadar gula darah puasa > 120 mg/dl menunjukkan kontribusi terhadap risiko penyakit.

-  **Fitur Kategorikal yang Relevan**:
  - `ExerciseAngina` – menunjukkan apakah pasien mengalami angina saat berolahraga.
  - `ChestPainType` – tipe nyeri dada berasosiasi erat dengan keberadaan penyakit jantung.
  - `ST_Slope` – kemiringan segmen ST setelah olahraga sangat terkait dengan kondisi jantung pasien.


  

---

## 4. Data Preparation

Tahapan data preparation dilakukan untuk mempersiapkan data sebelum masuk ke tahap pemodelan. Berikut langkah-langkah yang dilakukan:


### Penanganan Outlier

Outlier adalah nilai ekstrem yang berbeda jauh dari mayoritas data. Jika tidak ditangani, outlier dapat memengaruhi performa model.

- Dilakukan deteksi outlier menggunakan metode **IQR (Interquartile Range)**.
- Outlier **tidak dihapus**, namun **ditangani dengan teknik capping (pembatasan nilai)**:
  - **Batas bawah** = Q1 - 1.5 * IQR
  - **Batas atas** = Q3 + 1.5 * IQR
  - Nilai yang lebih kecil dari batas bawah diubah menjadi batas bawah.
  - Nilai yang lebih besar dari batas atas diubah menjadi batas atas.
- Teknik ini menjaga jumlah data tetap sama, sambil meminimalkan efek negatif dari nilai ekstrem.

Hasil dari penanganan outlier ini adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/5cd9beb0-7066-4551-846e-a7a52d4b0294)

---

### Encoding Variabel Kategorikal

Model machine learning hanya dapat memproses nilai numerik, sehingga variabel kategorikal perlu diubah menjadi bentuk angka.

- Digunakan teknik **One-Hot Encoding** untuk mengubah variabel kategorikal menjadi dummy variabel.
- Parameter `drop_first=True` digunakan untuk menghindari **dummy variable trap** (multikolinearitas antar variabel dummy).
- Hasil: setiap kategori dalam kolom dikonversi menjadi kolom baru dengan nilai 0 atau 1.
  
![image](https://github.com/user-attachments/assets/68352d33-1178-45f3-89ff-2265cb360bea)

---

### Split Data

- Data dibagi menjadi **training set (80%)** dan **testing set (20%)** menggunakan `train_test_split`.
- Parameter `random_state=42` digunakan agar hasil dapat direproduksi.

### Normalisasi Fitur (Khusus Model SVM)

Beberapa algoritma seperti **Support Vector Machine (SVM)** sensitif terhadap skala fitur.

- Digunakan **StandardScaler** untuk menormalkan fitur (mean = 0, std = 1).
- Normalisasi hanya dilakukan pada fitur numerik, dan hanya digunakan pada model SVM.

---

## 5. Modeling

Pada tahap ini, dilakukan proses pengembangan dan evaluasi model machine learning untuk memprediksi apakah seseorang berisiko mengalami penyakit jantung (*Heart Disease*) atau tidak. Dataset yang telah dibersihkan dan diencoding dibagi menjadi fitur (X) dan target (y), kemudian dipecah menjadi data pelatihan (training set) dan pengujian (testing set) dengan rasio 80:20.

Tiga algoritma pembelajaran mesin yang digunakan pada tahap awal pemodelan adalah:

---

### 1. Random Forest Classifier

**Random Forest** merupakan algoritma *ensemble* berbasis *decision tree* yang membangun banyak pohon keputusan secara paralel dan menggabungkan hasil prediksi dari setiap pohon menggunakan metode voting mayoritas. Pendekatan ini membuat Random Forest cukup tahan terhadap overfitting dan mampu menangani dataset dengan fitur yang kompleks dan jumlah yang besar.

- **Library**: `sklearn.ensemble.RandomForestClassifier`
- **Parameter utama**:
  - `random_state=42`: menjamin hasil yang konsisten di setiap eksekusi.

Model dilatih menggunakan data training dan diuji terhadap data testing. Evaluasi dilakukan dengan metrik *precision*, *recall*, dan *f1-score* dari `classification_report`.

---

### 2. Support Vector Machine (SVM)

**SVM** adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane optimal yang memisahkan data dari dua kelas secara maksimal. SVM sangat sensitif terhadap skala fitur, sehingga **normalisasi** sangat penting.

Sebelum model dilatih, dilakukan **standarisasi** fitur menggunakan `StandardScaler` dari `sklearn.preprocessing` agar fitur-fitur berada dalam skala distribusi yang sama.

- **Library**:
  - `sklearn.svm.SVC` untuk pemodelan,
  - `sklearn.preprocessing.StandardScaler` untuk normalisasi data.
- **Parameter utama**:
  - Default parameter (`kernel='rbf'`, `C=1.0`, dll.)
  - `random_state` tidak tersedia secara langsung di SVC, sehingga hasil bisa sedikit bervariasi jika data awal berubah.

Model kemudian dilatih pada data yang telah dinormalisasi, dan evaluasi dilakukan menggunakan data testing yang juga sudah dinormalisasi.

---
### 3. XGBoost (Extreme Gradient Boosting)

XGBoost adalah algoritma boosting yang kuat dan efisien, yang melatih pohon keputusan secara berurutan untuk memperbaiki kesalahan dari model sebelumnya. Sangat populer di kompetisi data science karena performanya yang tinggi.

- Library: xgboost.XGBClassifier
- Parameter:
  - use_label_encoder=False (nonaktifkan encoder lama)
  - eval_metric='logloss' (metrik evaluasi saat training)
  - random_state=42 untuk replikasi

Model dilatih pada data training dan diuji pada data testing. Evaluasi dilakukan dengan classification_report.

---
## Evaluation

Pada proyek ini, penilaian model menggunakan confusion matrix, akurasi, dan f1 score sebagai metrik evaluasi untuk masing-masing model. Akan dijelaskan terlebih dahulu bagaimana cara mendapatkan akurasi dan f1 score serta bagaimana cara menggunakan confusion matrix.

### Matriks Confusion, Akurasi, dan Skor f1

1. Matriks Confusion merupakan matriks yang menunjukkan jumlah prediksi benar dan salah untuk setiap kelas. Contoh dari Matriks Confusion beserta labelnya dapat dilihat pada gambar di bawah ini.

<img src="https://github.com/user-attachments/assets/0b200762-9005-4765-9924-8076faf96046" align="center"><br>
Formatnya:
[[TP, FP],
 [FN, TN]]

Terdapat 4 label pada matriks confusion seperti yang terlihat di gambar, yaitu TP, TN, FP, dan FN.
    a. *True Positive* (TP) merupakan jumlah data pada positif yang ditebak dengan benar.
    b. *True Negative* (TN) merupakan jumlah data pada negatif yang ditebak dengan benar.
    c. *False Positive* (FP) merupakan jumlah data yang ditebak dengan salah karena diprediksi positif, sedangkan aslinya adalah negatif.
    d. *False Negative* (FN) merupakan jumlah data yang ditebak dengan salah karena diprediksi negatif, sedangkan aslinya adalah positif.
    
2. Akurasi merupakan Persentase prediksi benar terhadap total prediksi.<br>
Formatnya:

<img src="https://github.com/user-attachments/assets/689a2934-4adb-42e8-b46a-e59f2e6b0508" align="center"><br>
4. Skor F1 merupakan rata-rata harmonik dari precision dan recall.
Formatnya:

<img src="https://github.com/user-attachments/assets/de176d91-a6b6-40a7-adc4-dd0d755eaa16" align="center"><br>
5. Precision merupakan proporsi prediksi positif yang benar-benar benar.<br>
Rumusnya:

<img src="https://github.com/user-attachments/assets/12b1ad68-e216-4bde-bce7-cfea6652e7e7" align="center"><br>
*Contoh*: Jika model memprediksi 10 data sebagai positif, tetapi hanya 7 yang benar-benar positif, maka precision adalah 7/10 = 0.7.

5. Recall (Sensitivity) merupakan proporsi data positif yang terdeteksi dengan benar oleh model.<br>
Rumusnya:

<img src="https://github.com/user-attachments/assets/2f6d9e5d-bd59-4999-84ae-9e85146e385c" align="center"><br>
*Contoh*: Jika model memprediksi 10 data sebagai positif, tetapi hanya 7 yang benar-benar positif, maka precision adalah 7/10 = 0.7.
 
### Penerapan Matriks Confusion, Akurasi, dan Skor f1
Berikut adalah ringkasan hasil evaluasi tiap model berdasarkan data uji:


###  1. Random Forest Classifier

Model Random Forest menunjukkan performa cukup stabil dan seimbang antar kelas.

- **Accuracy**: 0.86 (86%)
- **F1-Score**:
  - Kelas 0 (Tidak sakit): 0.84
  - Kelas 1 (Sakit): 0.88
- **Precision**:
  - Kelas 0: 0.82
  - Kelas 1: 0.89
- **Recall**:
  - Kelas 0: 0.82
  - Kelas 1: 0.87

📌 **Catatan**: Model ini memiliki *recall* yang tinggi pada kelas 1 (sakit), yang sangat penting dalam konteks deteksi penyakit. Namun, presisi untuk kelas 0 sedikit lebih rendah dibandingkan model lain.

---

###  2. Support Vector Machine (SVM)

Model SVM menghasilkan performa paling konsisten dan seimbang, bahkan mengungguli model lain secara umum.

- **Accuracy**: 0.87 (87%)
- **F1-Score**:
  - Kelas 0: 0.85
  - Kelas 1: 0.89
- **Precision**:
  - Kelas 0: 0.83
  - Kelas 1: 0.90
- **Recall**:
  - Kelas 0: 0.87
  - Kelas 1: 0.89

📌 **Catatan**: Dengan skor F1 tertinggi secara makro dan weighted average, model SVM merupakan kandidat kuat sebagai model terbaik dalam proyek ini. Ia berhasil menjaga keseimbangan yang baik antara deteksi pasien yang sakit dan tidak sakit.

---

###  3. Extreme Gradient Boosting (XGBoost)

Model XGBoost tampil cukup kompetitif, meskipun sedikit di bawah SVM dalam hal akurasi dan recall.

- **Accuracy**: 0.86 (86%)
- **F1-Score**:
  - Kelas 0: 0.84
  - Kelas 1: 0.86
- **Precision**:
  - Kelas 0: 0.81
  - Kelas 1: 0.90
- **Recall**:
  - Kelas 0: 0.87
  - Kelas 1: 0.85

📌 **Catatan**: Meskipun presisi untuk kelas 1 sangat tinggi (0.90), recall-nya sedikit lebih rendah daripada SVM dan Random Forest, yang bisa menyebabkan underdetection untuk pasien sakit.

---

### 🔍 Perbandingan Akurasi & F1-Score

| Model           | Akurasi | F1-Score (0) | F1-Score (1) | Macro F1 |
|----------------|---------|--------------|--------------|----------|
| Random Forest  | 0.86    | 0.84         | 0.88         | 0.86     |
| SVM            | 0.87    | 0.85         | 0.89         | 0.87     |
| XGBoost        | 0.86    | 0.84         | 0.86         | 0.86     |

---

## Kesimpulan
Berdasarkan hasil yang diperoleh setelah melakukan proses EDA dan pengujiaan model terbaik untuk peningkatan prestasi siswa dapat dismpulkan bahwah:

- Dataset yang digunakan mencakup berbagai parameter medis penting seperti usia, tekanan darah, kadar kolesterol, jenis nyeri dada, denyut jantung maksimum, dan kemiringan segmen ST, yang semuanya berperan dalam mendeteksi risiko penyakit jantung.

- Melalui proses *Exploratory Data Analysis (EDA)*, ditemukan bahwa fitur yang paling berpengaruh terhadap keberadaan penyakit jantung adalah:
  - **ChestPainType** (jenis nyeri dada)
  - **ST_Slope** (kemiringan segmen ST pada EKG)
  - **Oldpeak** (depresi ST setelah aktivitas)
  - **MaxHR** (denyut jantung maksimum)
  - **Age** dan **RestingBP** (tekanan darah istirahat)

- Tiga algoritma *machine learning* diterapkan: **Random Forest**, **Support Vector Machine (SVM)**, dan **XGBoost**, dengan hasil evaluasi sebagai berikut:
  - **SVM** memiliki performa terbaik dengan akurasi **87%** dan F1-score **0.87**
  - **Random Forest** dan **XGBoost** menunjukkan akurasi serupa (**86%**) dengan F1-score yang juga tinggi (**0.86**)

- Semua model dikembangkan melalui pipeline yang mencakup:
  - *Encoding* fitur kategorikal
  - *StandardScaler* untuk model SVM
  - Pembagian data latih dan uji
  - Evaluasi dengan metrik: **Accuracy**, **Precision**, **Recall**, dan **F1-score**

- Berdasarkan evaluasi model, **SVM** direkomendasikan sebagai model utama untuk prediksi penyakit jantung karena kinerjanya yang paling optimal.

- Sistem prediksi ini dapat digunakan sebagai *decision support tool* bagi tenaga medis, terutama untuk:
  - Deteksi dini risiko penyakit jantung
  - Skrining awal di fasilitas kesehatan dengan sumber daya terbatas
  - Prioritas penanganan pasien secara lebih efisien

- Proyek ini menunjukkan bahwa pendekatan berbasis data mampu memberikan **solusi prediktif yang efektif dan praktis** untuk membantu meningkatkan kualitas layanan kesehatan, khususnya dalam pencegahan dan penanganan penyakit jantung.



## Referensi
1. A. Alaa, M. van der Schaar. "Forecasting Individualized Disease Trajectories using Interpretable Deep Learning", Nature Communications, vol. 10, Article No. 4394, 2019.
DOI: 10.1038/s41467-019-11387-0
2. R. Detrano et al. "International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease", The American Journal of Cardiology, vol. 64, no. 5, pp. 304–310, 1989.
Dataset: Cleveland Heart Disease – UCI Machine Learning Repository, diakses pada 15 Mei 2025 dari https://archive.ics.uci.edu/ml/datasets/heart+Disease
3. D. Dua, C. Graff. "UCI Machine Learning Repository: Heart Disease Dataset". University of California, Irvine, School of Information and Computer Sciences.
Diakses pada 15 Mei 2025 dari https://archive.ics.uci.edu/ml/datasets/Heart+Disease
4. Jason Brownlee. "How to Develop a Heart Disease Prediction Model Using Machine Learning", Machine Learning Mastery, 2021.
Diakses pada 15 Mei 2025 dari https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
5. Dicoding Indonesia. "Belajar Machine Learning untuk Pemula". Dicoding Academy,
Diakses pada 15 Mei 2025 dari https://www.dicoding.com/academies/184




