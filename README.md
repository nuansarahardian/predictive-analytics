# Laporan Proyek Machine Learning - Nuansa Syafrie Rahardian

## Domain Proyek

Kesehatan mental merupakan aspek penting dalam kehidupan seseorang, yang memengaruhi cara berpikir, merasakan, dan bertindak dalam menghadapi berbagai situasi kehidupan. Dalam beberapa tahun terakhir, isu terkait kesehatan mental semakin mendapatkan perhatian, terutama karena meningkatnya tingkat stres, depresi, dan gangguan kecemasan yang dialami oleh individu dari berbagai kalangan. Meskipun demikian, masih banyak tantangan dalam upaya deteksi dini dan penanganan masalah kesehatan mental secara efektif.

Dengan memanfaatkan data yang tersedia dari berbagai faktor seperti usia, jenis kelamin, jam tidur, tingkat stres, kualitas diet, hingga kebiasaan merokok dan konsumsi alkohol, machine learning dapat digunakan untuk membangun model prediktif yang mampu mengidentifikasi individu dengan risiko gangguan kesehatan mental. Model ini diharapkan dapat membantu organisasi, institusi kesehatan, dan pembuat kebijakan dalam mengambil langkah preventif serta memberikan intervensi yang tepat sasaran.

Pada proyek ini, penulis menerapkan empat model pembelajaran mesin yaitu Random Forest, Naive Bayes, Support Vector Machine (SVM), dan Extreme Gradient Boosting (XGBoost). Pendekatan ini digunakan untuk membandingkan performa setiap model dalam memprediksi apakah seseorang memiliki gangguan kesehatan mental berdasarkan dataset yang terdiri dari 50.000 entri dan 17 atribut. Dataset ini mencakup informasi demografis, kebiasaan harian, dan faktor gaya hidup yang berkaitan erat dengan kesehatan mental. Dataset diperoleh dari https://zenodo.org/records/14838680

## Business Understanding

### Problem Statements

Berdasarkan latar belakang tersebut, berikut adalah pertanyaan-pertanyaan yang dijadikan fokus dalam proyek ini:

1. **Apakah usia dan jenis kelamin** (`Age`, `Gender`) **berpengaruh terhadap kemungkinan seseorang mengalami gangguan kesehatan mental (`Mental_Health_Condition`)?**
2. **Apakah tingkat stres, jam tidur, dan jam kerja** (`Stress_Level`, `Sleep_Hours`, `Working_Hours`) **memiliki korelasi signifikan dengan kondisi kesehatan mental (`Mental_Health_Condition`)?**
3. **Seberapa besar pengaruh aktivitas fisik, kualitas diet, dan penggunaan media sosial** (`Physical_Activity`, `Quality_of_Diet`, `Social_Media_Use`) **terhadap kesehatan mental (`Mental_Health_Condition`)?**
4. **Apakah kebiasaan merokok dan konsumsi alkohol** (`Smoking`, `Alcohol_Consumption`) **memengaruhi kondisi kesehatan mental (`Mental_Health_Condition`)?**
5. **Apakah seseorang yang memiliki riwayat konsultasi dengan profesional kesehatan mental** (`Seeking_Professional_Help`) **lebih cenderung memiliki gangguan kesehatan mental (`Mental_Health_Condition`)?**
6. **Faktor mana yang paling berpengaruh dalam menentukan kondisi kesehatan mental (`Mental_Health_Condition`) ketika mempertimbangkan semua atribut yang tersedia?**
7. **Apa model machine learning terbaik yang mampu memprediksi kondisi kesehatan mental secara akurat berdasarkan fitur-fitur yang tersedia?**


### Goals
Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini.
1. Menemukan pola-pola demografis dan gaya hidup yang berkorelasi dengan gangguan kesehatan mental.
2. Mengidentifikasi faktor risiko utama yang mempengaruhi kesehatan mental seseorang.
3. Menghasilkan model prediktif berbasis machine learning untuk mendeteksi kondisi kesehatan mental.
4. Membandingkan performa beberapa algoritma machine learning untuk menentukan model terbaik berdasarkan metrik evaluasi.
5. Memberikan wawasan data-driven yang dapat digunakan untuk mendukung kebijakan intervensi dini dalam bidang kesehatan mental.

   
### Solution Statement

1. Melakukan _Exploratory Data Analysis (EDA)_ untuk menggali wawasan dari data, termasuk pengaruh **usia** (`Age`), **gender** (`Gender`), **tingkat stres** (`Stress_Level`), **jam tidur** (`Sleep_Hours`), **jam kerja** (`Working_Hours`), **kebiasaan merokok** (`Smoking`), **konsumsi alkohol** (`Alcohol_Consumption`), dan faktor gaya hidup lainnya terhadap **kesehatan mental** (`Mental_Health_Condition`).

2. Menerapkan empat algoritma machine learning yaitu:
   - **Random Forest**
   - **Naive Bayes**
   - **Support Vector Machine (SVM)**
   - **Extreme Gradient Boosting (XGBoost)**

   untuk melakukan klasifikasi kondisi kesehatan mental seseorang berdasarkan data yang tersedia.

3. Menggunakan metrik evaluasi sebagai berikut untuk menilai performa model:
   - **Confusion Matrix**
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**

4. Menyusun interpretasi dan insight dari model terbaik yang dihasilkan, serta memberikan rekomendasi intervensi berbasis data berdasarkan faktor-faktor yang paling berpengaruh terhadap kondisi kesehatan mental.


## Data Understanding

Dataset yang digunakan dalam penelitian ini berjudul "Mental Health and Lifestyle Dataset for Sentiment Analysis" dan diperoleh dari platform Zenodo. Dataset ini dipublikasikan oleh Bhagwati Pandey pada tanggal 13 Juni 2024, dan dapat diakses melalui tautan berikut: https://zenodo.org/records/14838680. Dataset ini berisi 50.000 data individu dari berbagai negara, mencakup faktor-faktor demografis, kebiasaan gaya hidup, serta kondisi kesehatan mental.
Dataset ini sangat relevan untuk analisis kesehatan mental karena mengandung berbagai atribut penting seperti status kondisi kesehatan mental (Mental_Health_Condition), tingkat keparahan (Severity), riwayat konsultasi (Consultation_History), tingkat stres (Stress_Level), serta gaya hidup seperti kebiasaan tidur, olahraga, dan penggunaan media sosial. Dataset ini dirancang untuk digunakan dalam riset prediktif, analisis korelasi, dan studi perilaku.

Sample data yang terdapat pada dataset adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/e0b085f0-de49-43c2-a06b-a0e67b7066a1)

Infromasi dataset tersebut dapat dilihat pada gambar dibawah ini:

![image](https://github.com/user-attachments/assets/661082ca-e92c-4a18-8b9a-959654dae01b)


Dari gambar yang ditampilkan, terdapat 3 variabel bertipe float64, 4 variabel bertipe int64, dan 11 variabel bertipe object (kategorikal)
### Deskripsi Variabel
Dataset ini memiliki 17 variabel dengan keterangan sebagai berikut.
| No | Nama Atribut              | Tipe Data | Deskripsi                                                                 |
|----|---------------------------|-----------|--------------------------------------------------------------------------|
| 1  | User_ID                   | Integer   | ID unik untuk setiap individu dalam dataset. Tidak digunakan dalam analisis. |
| 2  | Age                       | Integer   | Usia individu dalam tahun.                                               |
| 3  | Gender                    | Object    | Identitas gender individu.                                               |
| 4  | Occupation                | Object    | Sektor pekerjaan dari individu.                                          |
| 5  | Country                   | Object    | Negara tempat tinggal individu.                                          |
| 6  | Mental_Health_Condition   | Object    | Menunjukkan apakah individu memiliki gangguan kesehatan mental.          |
| 7  | Severity                  | Object    | Tingkat keparahan dari gangguan kesehatan mental yang dialami.           |
| 8  | Consultation_History      | Object    | Menunjukkan apakah individu pernah berkonsultasi dengan profesional kesehatan mental. |
| 9  | Stress_Level              | Object    | Tingkat stres individu.                                                  |
| 10 | Sleep_Hours               | Float     | Rata-rata jumlah jam tidur individu per hari.                            |
| 11 | Work_Hours                | Integer   | Jumlah jam kerja individu per minggu.                                    |
| 12 | Physical_Activity_Hours   | Integer   | Jumlah jam aktivitas fisik individu per minggu.                          |
| 13 | Social_Media_Usage        | Float     | Jumlah jam penggunaan media sosial per hari oleh individu.               |
| 14 | Diet_Quality              | Object    | Kualitas pola makan individu.                                            |
| 15 | Smoking_Habit             | Object    | Kebiasaan merokok individu.                                              |
| 16 | Alcohol_Consumption       | Object    | Kebiasaan konsumsi alkohol individu.                                     |
| 17 | Medication_Usage          | Object    | Menunjukkan apakah individu menggunakan obat-obatan untuk kesehatan mental. |


### Menangani Missing Value dan Duplicate Data (Duplikasi Data)
Pada tahap ini kita akan mengecek data yang tidak valid pada dataset. Setelah diperiksa apakah terdapat kolom yang bernilai null, hasilnya adalah tidak ada. Sedangkan data duplikat atau data ganda juga tidak ada. Maka dengan demikian data siapa untuk dianalisis pada tahap selanjutnya.

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

Data yang berada di bawah atau di atas batas tersebut dianggap sebagai *outlier*. Dari hasil pengecekan, dapat disimpulkan bahwa tidak ditemukan outlier pada semua kolom numerik di dataset, dengan hasil sebagai beriut:
![Screenshot 2025-05-17 164354](https://github.com/user-attachments/assets/d3204598-6682-4de5-83f9-91dd64ef0589)



## Exploratory Data Analysis (EDA)

### Univariate Analysis EDA

Ada beberapa tahap yang akan kita lakukan, yakni:
Tahap pertama, membagi variabel-variabel menjadi 2 jenis, yaitu variabel numerikal dan variabel kategorikal. Berikut merupakan kolom-kolom yang termasuk dalam variabel numerikal maupun kategorikal. <br>

![image](https://github.com/user-attachments/assets/ea547cc3-6f73-49f0-94df-4d8a4bbc624c)
![image](https://github.com/user-attachments/assets/2b4fa296-e713-4798-ac02-a4b9c1b67b26)

Keterangan detail terkait kolom _object_ kategorikal dan _value_ nya dapat dilihat pada tabel berikut:
| **No** | **Kolom Kategorikal**     | **Jumlah Kategori Unik** | **Daftar Kategori**                                           |
| ------ | ------------------------- | ------------------------ | ------------------------------------------------------------- |
| 1      | Gender                    | 4                        | Male, Prefer not to say, Non-binary, Female                   |
| 2      | Occupation                | 7                        | Education, Engineering, Sales, IT, Healthcare, Finance, Other |
| 3      | Country                   | 7                        | Australia, Other, India, USA, Germany, Canada, UK             |
| 4      | Mental\_Health\_Condition | 2                        | Yes, No                                                       |
| 5      | Severity                  | 4                        | None, Low, Medium, High                                       |
| 6      | Consultation\_History     | 2                        | Yes, No                                                       |
| 7      | Stress\_Level             | 3                        | Low, Medium, High                                             |
| 8      | Diet\_Quality             | 3                        | Healthy, Unhealthy, Average                                   |
| 9      | Smoking\_Habit            | 4                        | Regular Smoker, Heavy Smoker, Non-Smoker, Occasional Smoker   |
| 10     | Alcohol\_Consumption      | 4                        | Regular Drinker, Social Drinker, Non-Drinker, Heavy Drinker   |
| 11     | Medication\_Usage         | 2                        | Yes, No                                                       |

Tahap kedua adalah menampilkan statistik ringkasan dataset, dengan hasil sebagai berikut dan insight yang diperoleh.
![image](https://github.com/user-attachments/assets/bf127a05-601f-4dfe-8dec-501341c4df72)

Penjelasan:
- Seluruh fitur memiliki jumlah observasi yang lengkap (tidak ada nilai yang hilang).
- Rata-rata jam kerja (55 jam/minggu) relatif tinggi, mengindikasikan beban kerja yang signifikan.
- Aktivitas fisik rata-rata (4.98 jam/minggu) mendekati rekomendasi WHO (≥150 menit).
- Penggunaan media sosial cukup tinggi (rata-rata 3.24 jam/hari), dapat menjadi indikator potensi pengaruh terhadap kondisi mental.

Insight Awal:
- Populasi berada dalam rentang usia produktif (18–65 tahun) dengan variasi yang lebar.
- Variasi yang besar pada jam kerja dan usia menunjukkan potensi perbedaan gaya hidup yang signifikan di antara responden.
- Beberapa individu tidak memiliki aktivitas fisik sama sekali (nilai minimum = 0).
- Korelasi antara fitur-fitur seperti jam kerja, aktivitas fisik, dan penggunaan media sosial dapat dianalisis lebih lanjut terhadap variabel stres atau kesehatan mental.

Tahap ketiga, Pada tahap ini, kita akan membuat visualisasi data kategorikal dalam bentuk grafik dengan menggunakan library python matplotlib dan seaborn. Hasilnya seperti gambar dibawah ini:


![download (20)](https://github.com/user-attachments/assets/f9b96af7-cfde-49fb-b1c3-df0ce85dec0a)
Interpretasi:
| Fitur                         | Analisis                                                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Gender**                    | Mayoritas responden adalah *Female* dan *Male*, sementara kategori *Non-binary* dan *Prefer not to say* relatif sedikit.     |
| **Occupation**                | Terdistribusi merata pada beberapa sektor, dengan "IT," "Healthcare," dan "Education" menjadi sektor dominan.                |
| **Country**                   | Distribusi negara relatif seimbang, dengan "India," "USA," dan "Germany" mendominasi.                                        |
| **Mental\_Health\_Condition** | Mayoritas tidak memiliki gangguan kesehatan mental (*No*), namun masih terdapat jumlah signifikan yang mengalaminya (*Yes*). |
| **Severity**                  | Kategori "None" sangat dominan dibandingkan kategori lainnya. Sementara "Low," "Medium," dan "High" relatif kecil.           |
| **Consultation\_History**     | Mayoritas belum pernah melakukan konsultasi profesional (*No*).                                                              |
| **Stress\_Level**             | Kebanyakan individu memiliki tingkat stres "Medium," diikuti oleh "Low" dan "High."                                          |
| **Diet\_Quality**             | Mayoritas memiliki diet yang "Average," diikuti oleh "Healthy" dan "Unhealthy."                                              |
| **Smoking\_Habit**            | Distribusi merata, namun "Non-Smoker" mendominasi.                                                                           |
| **Alcohol\_Consumption**      | "Non-Drinker" mendominasi, namun "Social Drinker" juga cukup tinggi.                                                         |
| **Medication\_Usage**         | Mayoritas tidak mengonsumsi obat-obatan kesehatan mental (*No*).                                                             |


Tahap keempat, kita akan membuat visualisasi data numerikal dalam bentuk grafik dengan menggunakan library python `matplotlib`. Hasilnya seperti gambar dibawah ini:

![download (19)](https://github.com/user-attachments/assets/8736df01-4c96-4f3e-84e0-a5c08cd3fd39)

Interprestasi:
| Fitur                         | Analisis                                                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **User\_ID**                  | Hanya merupakan identitas unik, tidak relevan untuk analisis.                                                  |
| **Age**                       | Distribusi usia cenderung merata dari usia 20 hingga 60 tahun.                                                 |
| **Sleep\_Hours**              | Kebanyakan individu tidur antara 5 hingga 8 jam per hari, yang merupakan durasi ideal. Pola yang cukup normal. |
| **Work\_Hours**               | Mayoritas bekerja antara 40 hingga 60 jam per minggu, mendekati standar kerja umum.                            |
| **Physical\_Activity\_Hours** | Mayoritas melakukan aktivitas fisik sekitar 2 hingga 6 jam per minggu.                                         |
| **Social\_Media\_Usage**      | Penggunaan media sosial rata-rata antara 1 hingga 4 jam per hari.                                              |



### Multivariate Analysis EDA

Pada bagian ini, akan ditunjukan hubungan antara dua variabel biasa disebut sebagai bivariate EDA. Selanjutnya, kita akan melakukan analisis data pada fitur kategori dan numerik.

#### 1. Ananlisis data pada fitur numerik StudyTimeWeekly (Waktu belajar setiap minggu) dengan GPA (Nilai Prestasi)

<img src="https://github.com/user-attachments/assets/e3460ee9-6449-4533-9b03-3e227749d36c" align="center"><br>
Interpretasi:
Siswa yang waktu belajaranya banyak mempengaruhi naiknya prestasi belajar(GPA).

#### 2. Ananlisis data pada fitur numerik Absences (Ketidakhadiran) dengan GPA (Nilai Prestasi)

<img src="https://github.com/user-attachments/assets/90ebe42b-3f7e-44f6-9f6d-ac41cf45c9ce" align="center"><br>
Interpretasi:
Absen(ketidakhadiran) siswa sangat mempengaruhi turun prestasinya(GPA).

#### 3. Ananlisis data pada fitur kategori Tutoring (Bimbingan Belajar) dengan GradeClass (Kategori Kelas)

<img src="https://github.com/user-attachments/assets/9ecf2453-805f-4757-96a5-fcf847e2ea06" align="center"><br>
Interpretasi:
Banyak siswa yang tidak mengikuti bimbingan belajar yang mendapat prestasi rendah (Grade F)

#### 4. Ananlisis data pada fitur kategori Genre (Jenis Kelamin) dengan GradeClass (Kategori Kelas)

<img src="https://github.com/user-attachments/assets/71e91698-39b4-43c7-8eec-de9e0a342a6f" align="center"><br>
Interpretasi:
Jenis kelamin pria lebih dominan memiliki prestasi lebih tinggi dibandingkan dengan wanita

#### 5. Ananlisis data pada fitur kategori kegiatan non akademik (Extracurricular, Sports, Music, Volunteering) dengan GPA (Nilai Prestasi)

<img src="https://github.com/user-attachments/assets/684468d1-1942-4065-91cd-1c0b83923b95" align="center"><br>
Interpretasi:
Lebih banyak siswa yang tidak mengikuti kegiatan ekstrakulikuler, olahraga dan musik mempengaruhi turunya nilai pretasi(GPA) mereka

#### 6. Ananlisis data pada fitur kategori ParentalSupport (Dukungan Orang Tua) dengan GradeClass (Kategori Kelas)

<img src="https://github.com/user-attachments/assets/c711088d-2d29-4056-8d0c-37d5e9737d19" align="center"><br>
Interpretasi:
Mayoritas dukungan orang tua sangat mempengaruhin nilai prestasi siswa (GPA). Semakin tinggi dukungan orang tua, maka semakin meningkat nilai prestasi dari anaknya.

#### 7. Melihat Korelasi Variabel dengan Menggunakan Heatmap

<img src="https://github.com/user-attachments/assets/be45973d-5230-4db7-8f2c-eab943ff641a" align="center"><br>
Interpretasi:

Nilai Prestasi Siswa memiliki
1. Korelasi negatif yang cukup kuat dengan ketidakhadiran(Absences).
2. Korelasi positif yang cukup lemah dengan waktu belajar setiap minggu(StudyTimeWeekly).

#### 8. Melihat Plot Scatter yang Memiliki Nilai Korelasi Positif dan Negatif

<img src="https://github.com/user-attachments/assets/a4e3ecb2-be4a-4408-9f91-982e8e4535bd" align="center"><br>
Interpretasi:

Nilai prestasi siswa (GPA) memiliki  korelasi negatif yang kuat pada ketidakhadiran (garis regresi menurun ke kanan bawah) dan korelatif positif cukup lemah pada waktu belajar setiap minggu (garis regresi naik ke kanan atas)

## Data Data Preparation

Pada tahap ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahap persiapan data perlu dilakukan, yaitu:
1. Drop kolom yang tidak digunakan dalam pemrosesan data
2. Encoding fitur kategori
3. Pembagian dataset dengan fungsi train_test_split dari library sklearn.

### Drop kolom yang tidak digunakan dalam pemrosesan data

Pada tahap ini ada beberapa kolom pada dataset yang tidak perlu digunakan dalam pemrosesan data yakni `StudentID`, `Ethnicity` dan `ParentalEducation`. Kolom-kolom ini akan dihapus menggunakan fungsi `drop()`. Hasilnya dapat dilihat pada gambar dibawah ini:

<img src="https://github.com/user-attachments/assets/69b9a96d-6baf-46da-988c-8a6897f65549" align="center"><br>
Hasilnya menampilkan variabel kolom `StudentID`, `Ethnicity` dan `ParentalEducation` sudah terhapus. Dataset siswa yang akan kita proses saat ini terdiri dari 2 tipe data float64, 2 tipe data int64 dan 8 tipe data objek(string).

### Encoding Fitur Kategori

Pada bagian ini, karena dataset fitur kategori kita sebelumnya sudah diubah dalam bentuk objek (string) pada tahap eksplorasi data analis maka kita perlu mengubah data kategori (yang berbentuk teks atau label) menjadi format numerik agar dapat diproses oleh algoritma machine learning. Encoding Fitur Kategorikal dilakukan 3 bagian, yakni:

1. *Label Encoding* yaitu, mengonversi nilai kategori menjadi angka integer (`0` dan `1`)). Variabel yang akan diproses yakni:  <br>
    a. *Tutoring* (Apakah siswa mengikuti bimbingan belajar?) <br>
    b. *Extracurricular* (Apakah siswa mengikuti kegiatan ektrakulikuler?) <br>
    c. *Sports* (Apakah siswa mengikuti kegiatan olahraga? <br>
    d. *Music* (Apakah siswa mengikuti kegiatan musik?) <br>
    e. *Volunteering* (Apakah siswa mengikuti kegiatan sukarelaan?)
2. *One Hot Ecoding* yaitu mengubah setiap kategori menjadi kolom biner terpisah untuk data tidak terurut). Variabel yang akan diproses yakni Gender.
3. *Ordinal Encoding* yaitu memberikan nilai integer berdasarkan hierarki atau urutan kategori). Variabel yang akan diproses yakni ParentalSupport.
Hasil setelah dilakukan data preprocessing dapat dilihat pada gambar berikut:

<img src="https://github.com/user-attachments/assets/632a0ef0-258b-4d2a-88d1-db70cc2c4a54" align="center"><br>

### Train-Test-Split
Langkah awal kita mengubah data objek ke data numeri dengan memanggil fungsi konversi objek to numerik. Selanjutnya, karena target kita adalah variabel GradeClass untuk mengetahui akurasi prediksi dari kategori kelas prestasi terbaik, maka kita akan membuang kolom tersebut dari data dan assign kolom tersebut ke variabel baru. Data training digunakan untuk melatih model dengan data yang ada, sedangkan data testing digunakan untuk menguji model yang dibuat menggunakan data yang belum dilatih. Pembagian data ini dilakukan dengan perbandingan 80% : 20% untuk data training dan data testing menggunakan train_test_split dari library sklearn. Berikut adalah data traning yang akan diproses (ditampilkan contoh 5 baris teratas):

<img src="https://github.com/user-attachments/assets/b9facf2a-f21c-48a4-96f4-ab6d08d915cb" align="center"><br>

Kemudian, kita melihat jumlah masing-masing *GradeClass* (Kategori Kelas) pada data testing untuk selanjutnya ditransformasikan menggunakan `LabelEncoder()`. `LabelEncoder()` berfungsi untuk memetakan setiap kategori unik dalam kolom *GradeClass* menjadi angka integer mulai dari `0`

## Modeling

Pada bagian ini, kita akan membangun 4 model machine learning untuk menguji sebarapa baik akurasi model, sehingga model tersebut yang disarankan untuk memprediksi prestasi siswa.

### 1. Model Development dengan Random Forest

Algoritma pembelajaran ensemble yang sangat populer untuk tugas klasifikasi dan regresi. Ini bekerja dengan membuat sejumlah pohon keputusan selama pelatihan dan menggabungkan hasilnya (melalui voting untuk klasifikasi atau rata-rata untuk regresi) untuk meningkatkan akurasi dan mengurangi overfitting.. <br>
    
Pada pemodelan ini, *Random Forest* diimplementasikan menggunakan `RandomForestClassifier` dari library `sklearn.ensemble` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `n_estimators` yaitu jumlah tree yang akan dibuat, `criterion` yaitu fungsi untuk menentukan kualitas *splitting data*, `max_depth` yaitu kedalaman maksimum setiap tree, dan `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah `n_estimators = 200`, `criterion = "entropy"`, `max_depth = 10`, `random_state = 50`.

### 2. Model Development dengan Extreme Gradient Boosting (XGBoost)

Algoritma Extreme Gradient Boosting merupakan salah satu algoritma boosting yang sangat kuat untuk tugas klasifikasi dan regresi. XGBoost dirancang untuk efisiensi, fleksibilitas, dan performa tinggi, serta sering digunakan dalam kompetisi machine learning. <br>

Pada pemodelan ini, XGBoost diimplementasikan menggunakan `XGBClassifier` dari library `xgboost` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `max_depth` yaitu kedalaman maksimum setiap tree, `n_estimators` yaitu jumlah tree yang akan dibuat, `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi, `learning rate` yaitu mengatur langkah setiap iterasi ketika meminimumkan *loss function*, dan `n_jobs` yaitu mengatur jumlah CPU threads untuk menjalankan XGBoost. Pada proyek ini, parameter yang digunakan adalah `max_depth = `6`, `n_estimators = 125`, `random_state = 30`, `learning_rate = 0.01`, `n_jobs = -1`.

### 3. Model Development dengan Support Vector Machine* (SVM)

Algoritman ini sangat efektif untuk klasifikasi dan regresi. SVM bekerja dengan mencari hyperplane optimal yang memisahkan data dalam ruang fitur, serta mendukung kernel untuk menangani data non-linear. <br>

Pada pemodelan ini, SVM diimplementasikan menggunakan `SVC` dari library `sklearn.svm` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `kernel` yaitu tipe kernel yang digunakan untuk mentransformasikan input data, `gamma` yaitu pengaruh dari sebuah contoh training, dan `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah `kernel = 'rbf'`, `gamma = 'auto'`, `random_state = 50`.

### 4. Model Development dengan Naive Bayes

Algoritman ini merupakan algoritma klasifikasi berbasis probabilistik yang didasarkan pada Teorema Bayes. Algoritma ini bekerja dengan asumsi bahwa semua fitur saling independen (meskipun dalam kenyataan sering tidak sepenuhnya demikian). <br>

Pada pemodelan ini, Naive Bayes diimplementasikan menggunakan `GaussianNB` dari library `sklearn.naive_bayes` karena datanya numerik dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter `var_smoothing` berfungsi menambahkan nilai kecil (`var_smoothing`) ke varians dari setiap fitur. Sedangkan Nilai `1e-9` adalah representasi ilmiah untuk angka `0.000000001` (atau `10⁻⁹`). Ini digunakan untuk menambahkan nilai kecil pada varians, sehingga tidak ada nilai varians yang terlalu kecil untuk menghasilkan masalah numerik.

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

#### 1. Model Development dengan Random Forest

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *Random Forest*

<img src="https://github.com/user-attachments/assets/2fefd179-4ab8-451b-bb4f-6ad86f5ee29d" align="center"><br>
Dari gambar di atas, terdapat 8 data yang diprediksi salah pada Grade A dan 14 data yang diprediksi salah pada Grade F. Diperoleh skor F1 nya adalah 0.93 dengan akurasi tepatnya adalah 0.9269 atau ≈92.69%.

#### 2. Model Development dengan XGBoots

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *XGBoots*

<img src="https://github.com/user-attachments/assets/6a28e4c0-af31-4ff7-b177-cdbdf09e3ba7" align="center"><br>
Dari gambar di atas, terdapat 5 data yang diprediksi salah pada Grade A dan 15 data yang diprediksi salah pada Grade F. Diperoleh skor F1 nya adalah 0.93 dengan akurasi tepatnya adalah 0.9332 atau ≈93.32%.

#### 3. Model Model Development dengan SVM

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *SVM*

<img src="https://github.com/user-attachments/assets/9958e5a6-ea7f-4618-9e03-0e5404e45e22" align="center"><br>
Dari gambar di atas, terdapat 16 data yang diprediksi salah pada Grade A dan 28 data yang diprediksi salah pada Grade F. Diperoleh skor F1 nya adalah 0.77 dengan akurasi tepatnya adalah 0.7808 atau ≈78.08%.

#### 4. Model Model Development dengan Naive Bayes

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *Naive Bayes*

<img src="https://github.com/user-attachments/assets/bd837e95-d081-46cd-a221-17f51ce7af33" align="center"><br>
Dari gambar di atas, terdapat 19 data yang diprediksi salah pada Grade A dan 13 data yang diprediksi salah pada Grade F. Diperoleh skor F1 nya adalah 0.79 dengan akurasi tepatnya adalah 0.7933 atau ≈79.33%.

### Hasil Evaluasi
Dari seluruh akurasi yang diketahui dari keempat model, dibentuk bar plot untuk melihat perbandingan nilai akurasi model sebagai berikut. 

<img src="https://github.com/user-attachments/assets/c0584047-c778-4d72-a451-b48e44764df5" align="center"><br>
Berdasarkan gambar di atas dan evaluasi masing-masing model untuk mengetahui skor akurasi, skor F1, dan jumlah kesalahan klasifikasi pada masing-masing model, didapat model *XGBoots* merupakan model terbaik karena memiliki skor akurasi dan skor F1 tertinggi, serta jumlah kesalahan klasifikasi yang paling sedikit, terutama pada Grade A. 

## Kesimpulan
Berdasarkan hasil yang diperoleh setelah melakukan proses EDA dan pengujiaan model terbaik untuk peningkatan prestasi siswa dapat dismpulkan bahwah:
1. Terdapat hubungan positif antara durasi belajar mingguan yang lebih tinggi dengan performa akademik (GPA dan GradeClass), namun tidak terlalu signifikan kenaikannya. Oleh karena itu, direkomendasikan waktu belajar berada diatas 20 jam per minggu.
2. Tingkat absensi yang tinggi secara konsisten menunjukkan korelasi negatif dengan performa akademik. Siswa dengan absensi tinggi cenderung memiliki GPA lebih rendah. Oleh karena iu, perlu diidentifikasi siswa dengan pola absensi tinggi untuk intervensi dini.
3. Siswa yang tidak mengikuti bimbingan belajar menunjukkan GPA yang lebih rendah dibandingkan mereka yang mengikuti. Olehkarena itu siswa perlu disiplin mengikuti bimbingan belajar.
4. Jenis kelamin yang berbeda memiliki prestasi yang tidak jauh berbeda. Dari data yang diperoleh, mayoritas perempuan mengalami penurunan prestasi pada Grade F, Grade D, Grade B dan Grade A dibandingkan dengan laki-laki.
5. Terjadi penurunan sedikit nilai pretasinya(GPA) siswa yang tidak mengikuti kegiatan ekstrakulikuler, olahraga dan musik. Sedangkan untuk kegiatan sukarelaan terlihat merata atau seimbang. 
6. Siswa yang menerima dukungan orang tua memiliki nilai rata-rata lebih tinggi dibandingkan yang tidak.
7. *Exploratory Data Analysis* (EDA) menunjukkan bahwa performa akademik siswa dipengaruhi oleh kombinasi faktor internal (seperti durasi belajar dan pola absensi) serta faktor eksternal (seperti keterlibatan orang tua dan partisipasi dalam kegiatan non-akademik)
8. Setelah menguji data menggunakan 4 model *machine learning*, yaitu ***Extreme Gradient Boosting* (XGBoost)**, ***Support Vector Machine* (SVM)**, ***Naive Bayes*** dan ***Random Forest*** untuk memprediksi performa siswa, diperoleh:
    * ***XGBoost*** adalah model terbaik untuk memprediksi performa siswa pada dataset ini, dengan akurasi dan F1-Score tertinggi.
    * ***Random Forest*** memberikan hasil yang hampir setara dengan XGBoost dan lebih mudah diimplementasikan.
    * ***SVM*** memberikan performa baik tetapi memerlukan penyesuaian parameter untuk hasil optimal.
    * ***Naive Bayes*** adalah model tercepat namun memiliki performa yang jauh lebih rendah karena asumsi independensi antar fitur yang tidak sesuai dengan dataset.

## Referensi
1. Abdul Rahman. "Klasifikasi Performa Akademik Siswa Menggunakan Metode Decision Tree dan Naive Bayes", Vol. 13 No.1 (2023) 22-31, ISSN 2503-3247. SINTA Peringkat 4, diakses pada 28 November 2024.
2. Dicoding. Diakses pada 6 Juli 2024 dari https://www.dicoding.com/academies/319-machine-learning-terapan
3. Arif Fahrudin1, Harco Leslie Hendric Spits Warnars. "Prediksi Performa Siswa Dengan Metode SAW", vol. 9, no. 1, 2020, P-ISSN 2089-1245, E-ISSN 2655-4925. KILAT, diakses pada 29 November 2024.
![dos-b06f71539f3fe1ba6cd72f2403a77d1320240620141051](https://github.com/user-attachments/assets/70390d4c-e501-4a9b-8137-05d55cea8ea8)

<img src="https://github.com/user-attachments/assets/70390d4c-e501-4a9b-8137-05d55cea8ea8" align="center"><br>



