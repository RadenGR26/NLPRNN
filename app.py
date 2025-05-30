import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- Konstanta Aplikasi ---
# PASTIKAN NAMA FILE DAN LOKASI INI SESUAI DENGAN DATASET ANDA
DATA_PATH = 'data/dataset.csv' # Menggunakan dataset.csv sesuai konfirmasi Anda
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'rnn_model.h5')
MAX_LEN = 100         # Panjang maksimum sequence teks
EMBEDDING_DIM = 64    # Dimensi embedding untuk layer Embedding
RNN_UNITS = 32        # Jumlah unit pada layer SimpleRNN
EPOCHS = 10           # Jumlah epoch untuk pelatihan model (ditingkatkan sedikit)
BATCH_SIZE = 32       # Ukuran batch untuk pelatihan model

# --- Persiapan Direktori ---
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# --- Variabel Global untuk Model dan Tokenizer ---
# Ini akan diisi setelah data dimuat atau model dilatih/dimuat
global tokenizer, model, vocab_size
tokenizer = None
model = None
vocab_size = 1 # Default, akan diperbarui jika data tersedia

# =============================================== #
# ==== 1. Fungsi Pemuatan dan Persiapan Data ==== #
# =============================================== #
def load_and_prepare_data():
    """
    Memuat dataset dari CSV, membersihkan kolom, dan melakukan mapping label.
    Mengembalikan DataFrame yang siap digunakan.
    """
    print(f"🔄 Mencoba memuat dataset dari: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()

        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ Kolom yang dibutuhkan ('text', 'label') tidak ditemukan.")
            print(f"   Kolom yang tersedia: {df.columns.tolist()}")
            return pd.DataFrame(columns=required_columns)

        # Pilih kolom yang dibutuhkan dan hapus baris yang memiliki NaN di kedua kolom ini
        df = df[required_columns].dropna()
        df['text'] = df['text'].astype(str)

        # --- DEBUGGING LINES START ---
        # Pastikan label diproses dengan benar (trim spasi & lowercase) sebelum mapping
        df.loc[:, 'label'] = df['label'].astype(str).str.strip().str.lower()
        print(f"DEBUG: Unique labels setelah strip & lowercase: {df['label'].unique().tolist()}")
        # --- DEBUGGING LINES END ---

        # Mapping label string ke angka
        label_map = {'positif': 1, 'negatif': 0}
        df.loc[:, 'label'] = df['label'].map(label_map)

        # --- DEBUGGING LINES START ---
        print(f"DEBUG: Unique labels setelah mapping: {df['label'].unique().tolist()}")
        # --- DEBUGGING LINES END ---

        # Hapus baris label yang tidak dikenali (NaN setelah mapping)
        # Baris ini penting karena jika ada label selain 'positif'/'negatif',
        # mereka akan menjadi NaN setelah mapping dan harus dihapus.
        df = df.dropna(subset=['label'])
        df.loc[:, 'label'] = df['label'].astype(int) # Pastikan tipe datanya int di DataFrame

        if df.empty:
            print("⚠️ Dataset dimuat tetapi kosong setelah preprocessing. Periksa isi file CSV Anda.")
        else:
            print(f"✅ Dataset berhasil dimuat dengan {len(df)} baris data.")
        return df

    except FileNotFoundError:
        print(f"❌ ERROR: File dataset tidak ditemukan di: {DATA_PATH}.")
        print("   Pastikan file 'dataset.csv' ada di folder 'data/'.")
        return pd.DataFrame(columns=required_columns)
    except Exception as e:
        print(f"❌ ERROR: Gagal memuat atau mempersiapkan dataset: {e}")
        return pd.DataFrame(columns=required_columns)

# =============================================== #
# ==== 2. Fungsi Pembentukan Model RNN ==== #
# =============================================== #
def build_rnn_model(vocab_size_param):
    """
    Membangun arsitektur model Simple RNN.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size_param, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        SimpleRNN(RNN_UNITS, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Model RNN berhasil dibangun.")
    return model

# =============================================== #
# ==== 3. Inisialisasi Data, Tokenizer, Model ==== #
# =============================================== #
def initialize_assets():
    """
    Menginisialisasi tokenizer dan model (memuat atau melatih).
    Fungsi ini dipanggil sekali saat aplikasi dimulai.
    """
    global tokenizer, model, vocab_size

    df = load_and_prepare_data()

    # Inisialisasi Tokenizer
    print("🔄 Menginisialisasi Tokenizer...")
    tokenizer = Tokenizer()
    if not df.empty:
        tokenizer.fit_on_texts(df['text'])
        vocab_size = len(tokenizer.word_index) + 1
        print(f"✅ Tokenizer berhasil diinisialisasi. Ukuran vocabulary: {vocab_size}")
    else:
        print("⚠️ Dataset kosong, tokenizer diinisialisasi dengan vocabulary minimal (vocab_size=1).")
        vocab_size = 1 # Biarkan default, model akan dibangun dengan ini

    # Inisialisasi Model
    print("🔄 Mempersiapkan model (memuat atau melatih)...")
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ ERROR: Gagal memuat model dari {MODEL_PATH}: {e}")
            print("   Membangun dan melatih model baru sebagai gantinya.")
            model = build_rnn_model(vocab_size)
            if not df.empty:
                train_and_save_model(df, tokenizer, model)
            else:
                print("⚠️ Dataset kosong, model tidak dapat dilatih. Hanya model dasar yang dibuat.")
    else:
        print(f"⚠️ Model tidak ditemukan di {MODEL_PATH}.")
        print("   Membangun dan melatih model baru.")
        model = build_rnn_model(vocab_size)
        if not df.empty:
            train_and_save_model(df, tokenizer, model)
        else:
            print("⚠️ Dataset kosong, model tidak dapat dilatih. Hanya model dasar yang dibuat.")

    # Pastikan model terdefinisi, jika gagal total, buat model dummy
    if model is None:
        print("❌ ERROR: Model gagal diinisialisasi atau dimuat. Membuat model dummy.")
        model = build_rnn_model(vocab_size)


def train_and_save_model(df, tokenizer_obj, model_obj):
    """
    Melatih model dan menyimpannya.
    """
    sequences = tokenizer_obj.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

    # --- PERBAIKAN UNTUK Invalid dtype: object ---
    # Pastikan padded_sequences adalah NumPy array dengan dtype yang benar
    padded_sequences = np.array(padded_sequences, dtype='int32')

    labels = df['label'].values
    # Pastikan labels adalah NumPy array dengan dtype yang benar
    labels = np.array(labels, dtype='int32') # Labels binary (0 atau 1) cocok dengan int32
    # --- AKHIR PERBAIKAN ---

    if len(padded_sequences) < 2 or len(np.unique(labels)) < 2:
        print("❌ Tidak cukup data atau hanya satu kelas untuk melakukan train-test split. Model tidak dilatih.")
        print(f"   Jumlah sampel: {len(padded_sequences)}, Jumlah kelas unik: {len(np.unique(labels))}")
        return

    # Pastikan data memiliki setidaknya dua kelas untuk split
    if len(np.unique(labels)) < 2:
        print("❌ Hanya ditemukan satu kelas label di dataset. Train-test split membutuhkan setidaknya dua kelas.")
        print("   Model tidak dilatih.")
        return

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)

    print("⏳ Mulai melatih model...")
    try:
        model_obj.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
        model_obj.save(MODEL_PATH)
        print(f"✅ Model berhasil dilatih dan disimpan ke: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Gagal melatih atau menyimpan model: {e}")


# Panggil inisialisasi saat aplikasi Flask dimulai
with app.app_context():
    initialize_assets()

# =============================================== #
# ==== 4. Fungsi Prediksi Teks ==== #
# =============================================== #
def predict_sentiment(text_input):
    """
    Melakukan prediksi sentimen pada teks input.
    """
    if tokenizer is None or model is None:
        print("❌ ERROR: Tokenizer atau Model belum diinisialisasi saat prediksi.")
        return "Error: Sistem belum siap (model/tokenizer tidak ada)."

    seq = tokenizer.texts_to_sequences([text_input])
    # Pastikan padded memiliki setidaknya satu dimensi
    if not seq or not seq[0]: # Jika teks tidak dikenal oleh tokenizer, seq[0] bisa kosong
        print(f"⚠️ Peringatan: Teks '{text_input}' tidak menghasilkan sequence token yang valid.")
        # Mengembalikan prediksi netral atau error
        return "Tidak Diketahui" # Atau bisa juga "Error: Teks tidak dikenali"

    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    try:
        prediction = model.predict(padded)[0][0]
        sentiment = "Positif" if prediction >= 0.5 else "Negatif"
        print(f"➡️ Teks: '{text_input}' -> Prediksi: {prediction:.4f} ({sentiment})")
        return sentiment
    except Exception as e:
        print(f"❌ ERROR: Gagal melakukan prediksi untuk teks '{text_input}': {e}")
        return "Error: Gagal memprediksi."


# =============================================== #
# ==== 5. Rute Flask UI ==== #
# =============================================== #
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Rute utama untuk halaman web. Menangani tampilan dan prediksi sentimen.
    """
    result = None
    user_text = "" # Inisialisasi untuk menjaga teks di textarea

    if request.method == 'POST':
        user_text = request.form['text']
        if user_text:
            result = predict_sentiment(user_text)
        else:
            result = "Silakan masukkan teks untuk dianalisis."

    return render_template('index.html', result=result, user_text=user_text)

# =============================================== #
# ==== 6. Jalankan Aplikasi Flask ==== #
# =============================================== #
if __name__ == '__main__':
    # Sebelum menjalankan, pastikan struktur folder:
    # your_project_folder/
    # ├── app.py
    # ├── index.html
    # ├── data/
    # │   └── dataset.csv  <-- Pastikan file ini ada dengan konten yang benar
    # └── model/ (akan dibuat otomatis jika belum ada atau model dilatih)

    print("\n--- Aplikasi Flask Siap Dijalankan ---")
    print(f"Akses aplikasi di: http://127.0.0.1:5000/")
    print("--------------------------------------\n")
    app.run(debug=True) # debug=True akan otomatis me-restart server saat ada perubahan kode