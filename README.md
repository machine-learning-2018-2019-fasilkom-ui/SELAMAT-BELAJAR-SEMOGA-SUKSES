# CerpenG - Pendekatan Machine Learning untuk Prediksi Genre Cerpen

Firman Hadi Prayoga - 1606862721

Norman Bintang - 1606862772

Windi Chandra - 1606862785

Fakultas Ilmu Komputer - Universitas Indonesia

### Abstrak
Untuk pengalaman pengguna yang lebih baik, kami membuat sistem CerpenG yang dapat memprediksi sebuah genre dari suatu
cerpen berbahasa Indonesia. Sistem kami juga dapat menangani cerpen yang memiliki lebih dari
satu genre. Kami juga menunjukkan bahwa sistem kami yang diimplementasi menggunakan
SVM (*Support Vector Machine*) mempunyai performa prediksi yang lebih baik dibandingkan dengan implementasi
*Multinomial Naive Bayes*.


### Menjalankan Program
- Pastikan Anda mempunyai Python versi 3 dan
seluruh *dependency* yang ada di berkas `requirements.txt` sudah di-*install*.

- Model yang sudah *pre-trained* (Doc2Vec, Multilabel MNB, Multilabel SVM) dapat diunduh di
[tautan ini](https://drive.google.com/file/d/1rdywcrDqJWguBP4RCnGc1eDNpvWGVsEm/view?usp=sharing).
Isi dari arsip silakan dimasukkan ke dalam *folder* `pickled_models`.

- Apabila ingin memulai *training* dari awal, dapat menghapus tanda komentar pada bagian yang sesuai dengan
model yang ingin dilakukan *training*. (Perhatian: **cukup lama**)

- Terdapat dua buah *script* yang bisa dijalankan:
`naive_bayes_main.py` (*baseline model*), dan
`svm_classifier_main.py` (*SVM model*). Kedua *script* akan melakukan prediksi
dan evaluasi dan menyimpan hasilnya pada folder `results`.

### Test Program
- Terdapat sejumlah rangkaian tes untuk beberapa model pada *folder*
`tests`. Rangkaian tes tersebut diambil dari kasus uji pada PR 1 (*Naive Bayes*)
 dan PR 2 (SVM).