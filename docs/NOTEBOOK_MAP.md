# Peta notebook — `notebooks/btc_walkforward_3model.ipynb`

**Digenerate oleh `tools/notebook_map.py`. Jangan disunting tangan** — jalankan ulang setelah notebook berubah.

Notebook ini adalah deliverable utama: setiap modul `src/itransformer_btc/` menjadi satu sel definisi, salinan kode resmi thuml/iTransformer ditulis oleh sel `%%writefile`, dan sel langkah menjalankan studi. `src/` adalah proyeksinya yang diuji (`tests/test_notebook.py`), sehingga peta ini menyebut modul alih-alih mengulang kodenya.

- Sel: **90** (41 kode, 49 markdown)
- Fase: **12**
- Langkah produksi (`role: step`): **16**
- Modul yang diproyeksikan: **17** dalam 17 sel definisi

## Peta artefak — sel mana menghasilkan apa

Tiap sel langkah mendeklarasikan berkas yang dibaca dan ditulisnya.

| Langkah | Sel | Fase | Membaca | Menulis |
| --- | --: | --- | --- | --- |
| `artifact_map` | 3 | 🔧 01 · Persiapan lingkungan dan konfigurasi | — | — |
| `setup` | 5 | 🔧 01 · Persiapan lingkungan dan konfigurasi | `data/raw/BTCUSDT_1h.parquet` | — |
| `data` | 20 | 📥 02 · Muat data dan audit kualitas | `data/raw/BTCUSDT_1h.parquet` | — |
| `features` | 25 | 🧪 03 · Feature engineering dan eksplorasi | — | — |
| `keff` | 34 | 🪟 04 · Split walk-forward, scaling, dan K_eff | — | `artifacts/keff_table.parquet` |
| `upstream` | 53 | 🧠 05 · Model, baseline, dan fungsi training | `vendor/thuml_iTransformer/**` | — |
| `code_digest` | 70 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | — | — |
| `invariants` | 72 | 🛠️ 07 · Pemeriksaan sebelum training | — | `artifacts/naive_rw_by_origin.parquet` |
| `pilot` | 74 | 🛡️ 08 · Validasi pilot dan rencana sesi | — | `artifacts/validation/*.json`, `artifacts/pilot.json` |
| `grid` | 76 | 🚀 09 · Training grid walk-forward | `data/raw/BTCUSDT_1h.parquet` | `artifacts/preds/*.parquet`, `artifacts/meta/*.json`, `artifacts/weights/*.pt`, `artifacts/checkpoints/*.pt`, `artifacts/session_status.json` |
| `evaluate` | 79 | 📈 10 · Evaluasi model dan research questions | `artifacts/preds/*.parquet`, `artifacts/meta/*.json` | — |
| `contrasts` | 81 | 📈 10 · Evaluasi model dan research questions | — | — |
| `research_questions` | 83 | 📈 10 · Evaluasi model dan research questions | — | — |
| `direction` | 85 | 📈 10 · Evaluasi model dan research questions | — | — |
| `report` | 87 | 💾 11 · Simpan hasil, tabel, dan figure | — | `paper/paper_numbers.json`, `paper/tables/*.tex`, `paper/figures/*.pdf`, `paper/figures/*.png`, `paper/panels/*.parquet` |
| `sync_back` | 89 | 🔁 12 · Lampiran — sinkronisasi lokal | — | — |

## Modul yang dibawa notebook

Satu sel definisi per modul, dalam urutan eksekusi.

| Modul | Sel | Fase |
| --- | --: | --- |
| `src/itransformer_btc/config.py` | 9 | 🔧 01 · Persiapan lingkungan dan konfigurasi |
| `src/itransformer_btc/__init__.py` | 11 | 🔧 01 · Persiapan lingkungan dan konfigurasi |
| `src/itransformer_btc/segments.py` | 14 | 📥 02 · Muat data dan audit kualitas |
| `src/itransformer_btc/windows.py` | 16 | 📥 02 · Muat data dan audit kualitas |
| `src/itransformer_btc/budget.py` | 18 | 📥 02 · Muat data dan audit kualitas |
| `src/itransformer_btc/features.py` | 23 | 🧪 03 · Feature engineering dan eksplorasi |
| `src/itransformer_btc/efficiency.py` | 27 | 🧪 03 · Feature engineering dan eksplorasi |
| `src/itransformer_btc/splits.py` | 30 | 🪟 04 · Split walk-forward, scaling, dan K_eff |
| `src/itransformer_btc/keff.py` | 32 | 🪟 04 · Split walk-forward, scaling, dan K_eff |
| `src/itransformer_btc/upstream.py` | 37 | 🧠 05 · Model, baseline, dan fungsi training |
| `src/itransformer_btc/model.py` | 55 | 🧠 05 · Model, baseline, dan fungsi training |
| `src/itransformer_btc/train.py` | 57 | 🧠 05 · Model, baseline, dan fungsi training |
| `src/itransformer_btc/baselines.py` | 59 | 🧠 05 · Model, baseline, dan fungsi training |
| `src/itransformer_btc/metrics.py` | 62 | ⚙️ 06 · Persiapan evaluasi dan eksekutor |
| `src/itransformer_btc/comparisons.py` | 64 | ⚙️ 06 · Persiapan evaluasi dan eksekutor |
| `src/itransformer_btc/runner.py` | 66 | ⚙️ 06 · Persiapan evaluasi dan eksekutor |
| `src/itransformer_btc/report.py` | 68 | ⚙️ 06 · Persiapan evaluasi dan eksekutor |

## Salinan kode upstream

Ditulis apa adanya oleh `%%writefile`, lalu diverifikasi terhadap sha256 yang dipin.

| Berkas | Sel |
| --- | --: |
| `vendor/thuml_iTransformer/LICENSE` | 39 |
| `vendor/thuml_iTransformer/layers/Embed.py` | 41 |
| `vendor/thuml_iTransformer/layers/SelfAttention_Family.py` | 43 |
| `vendor/thuml_iTransformer/layers/Transformer_EncDec.py` | 45 |
| `vendor/thuml_iTransformer/utils/masking.py` | 47 |
| `vendor/thuml_iTransformer/model/iTransformer.py` | 49 |
| `vendor/thuml_iTransformer/model/Transformer.py` | 51 |

## Fase

### 🔧 01 · Persiapan lingkungan dan konfigurasi

Sesi Kaggle GPU T4 × 2, dataset input, impor bersama, dan konfigurasi penelitian.

- **Langkah `artifact_map`** (sel 3) — 🗺️ Peta artefak
- **Langkah `setup`** (sel 5) — 🧰 Setup sesi · membaca `data/raw/BTCUSDT_1h.parquet`
- `library` (sel 7) — 📚 Library
- Modul `src/itransformer_btc/config.py` (sel 9)
- Modul `src/itransformer_btc/__init__.py` (sel 11)

### 📥 02 · Muat data dan audit kualitas

Gap memutus deret tanpa imputasi; jendela divalidasi lewat timestamp; anggaran jendela dicocokkan per origin.

- Modul `src/itransformer_btc/segments.py` (sel 14)
- Modul `src/itransformer_btc/windows.py` (sel 16)
- Modul `src/itransformer_btc/budget.py` (sel 18)
- **Langkah `data`** (sel 20) — 📊 Muat bar dan audit anggaran · membaca `data/raw/BTCUSDT_1h.parquet`

### 🧪 03 · Feature engineering dan eksplorasi

Dua belas variat per-bar F1–F5 tanpa rolling window, dan diagnostik efisiensi pasar.

- Modul `src/itransformer_btc/features.py` (sel 23)
- **Langkah `features`** (sel 25) — 🔬 Bangun frame fitur
- Modul `src/itransformer_btc/efficiency.py` (sel 27)

### 🪟 04 · Split walk-forward, scaling, dan K_eff

21 bulan latih dan 3 bulan validasi, purge 24 jam di kedua batas, scaler dari data latih saja, K_eff sebelum training.

- Modul `src/itransformer_btc/splits.py` (sel 30)
- Modul `src/itransformer_btc/keff.py` (sel 32)
- **Langkah `keff`** (sel 34) — 📐 Ukur K_eff · menulis `artifacts/keff_table.parquet`

### 🧠 05 · Model, baseline, dan fungsi training

Kode arsitektur disalin dari thuml/iTransformer pada commit terkunci dan diverifikasi sha256; Ridge dari scikit-learn; satu loop latih untuk kedua Transformer.

- Modul `src/itransformer_btc/upstream.py` (sel 37)
- Salinan upstream `LICENSE` (sel 39)
- Salinan upstream `layers/Embed.py` (sel 41)
- Salinan upstream `layers/SelfAttention_Family.py` (sel 43)
- Salinan upstream `layers/Transformer_EncDec.py` (sel 45)
- Salinan upstream `utils/masking.py` (sel 47)
- Salinan upstream `model/iTransformer.py` (sel 49)
- Salinan upstream `model/Transformer.py` (sel 51)
- **Langkah `upstream`** (sel 53) — 🔐 Verifikasi dan muat kode upstream · membaca `vendor/thuml_iTransformer/**`
- Modul `src/itransformer_btc/model.py` (sel 55)
- Modul `src/itransformer_btc/train.py` (sel 57)
- Modul `src/itransformer_btc/baselines.py` (sel 59)

### ⚙️ 06 · Persiapan evaluasi dan eksekutor

Metrik, perbandingan antarmodel, manifes 900 run, eksekutor dua GPU, dan laporan; digest kode dicatat sebelum training.

- Modul `src/itransformer_btc/metrics.py` (sel 62)
- Modul `src/itransformer_btc/comparisons.py` (sel 64)
- Modul `src/itransformer_btc/runner.py` (sel 66)
- Modul `src/itransformer_btc/report.py` (sel 68)
- **Langkah `code_digest`** (sel 70) — 🔏 Provenance kode

### 🛠️ 07 · Pemeriksaan sebelum training

Invarian skala use_norm, overfit satu batch untuk kedua Transformer, jumlah parameter, dan Naive-RW per origin.

- **Langkah `invariants`** (sel 72) · menulis `artifacts/naive_rw_by_origin.parquet`

### 🛡️ 08 · Validasi pilot dan rencana sesi

Tiga model × empat K pada validation origin pertama, satu run per GPU: cek teknis dan waktu, tanpa memilih apa pun; lalu rencana sesi dan digest desain untuk dibekukan.

- **Langkah `pilot`** (sel 74) · menulis `artifacts/validation/*.json`, `artifacts/pilot.json`

### 🚀 09 · Training grid walk-forward

900 run, satu worker per GPU, resume otomatis dari output sesi sebelumnya; berjalan hanya setelah digest desain dibekukan di sel ini.

- **Langkah `grid`** (sel 76) · menulis `artifacts/preds/*.parquet`, `artifacts/meta/*.json`, `artifacts/weights/*.pt`, `artifacts/checkpoints/*.pt`, `artifacts/session_status.json` · membaca `data/raw/BTCUSDT_1h.parquet`

### 📈 10 · Evaluasi model dan research questions

C1 (iTransformer vs Ridge), C2 (iTransformer vs Transformer), RQ1, RQ2, dan akurasi arah dari prediksi tersimpan. Semua inferensi bersifat diagnostik: origin berbagi data latih.

- **Langkah `evaluate`** (sel 79) — 📋 Hasil utama · membaca `artifacts/preds/*.parquet`, `artifacts/meta/*.json`
- **Langkah `contrasts`** (sel 81) — ⚖️ C1 dan C2
- **Langkah `research_questions`** (sel 83) — 🔢 RQ1 dan RQ2
- **Langkah `direction`** (sel 85) — 🧭 Akurasi arah

### 💾 11 · Simpan hasil, tabel, dan figure

paper_numbers.json, tabel LaTeX, figure, dan panel, semuanya dari prediksi tersimpan.

- **Langkah `report`** (sel 87) · menulis `paper/paper_numbers.json`, `paper/tables/*.tex`, `paper/figures/*.pdf`, `paper/figures/*.png`, `paper/panels/*.parquet`

### 🔁 12 · Lampiran — sinkronisasi lokal

Menulis src/ dari sel definisi notebook yang sudah disimpan; nonaktif, hanya untuk checkout lokal.

- **Langkah `sync_back`** (sel 89)
