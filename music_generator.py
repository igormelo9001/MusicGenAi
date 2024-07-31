import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import BinaryCrossentropy

# Funções de Pré-processamento e Modelo
def load_and_preprocess_audio(file_paths):
    data = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        data.append(S_dB.T)
    return data

def create_sequences(data, seq_length):
    X = []
    for song in data:
        num_sequences = len(song) // seq_length
        for i in range(num_sequences):
            start = i * seq_length
            end = start + seq_length
            X.append(song[start:end])
    return np.array(X)

def build_gan(input_shape):
    # Gerador
    def build_generator():
        model = models.Sequential([
            layers.Input(shape=(100,)),
            layers.Dense(input_shape[0] // 2 * input_shape[1] * 16, activation='relu'),
            layers.Reshape((input_shape[0] // 2, input_shape[1] * 16)),
            layers.Conv1DTranspose(128, kernel_size=5, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(64, kernel_size=5, activation='relu', padding='same'),
            layers.Conv1DTranspose(input_shape[1], kernel_size=5, activation='sigmoid', padding='same')
        ])
        return model

    # Discriminador
    def build_discriminator():
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, kernel_size=5, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(128, kernel_size=5, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    generator = build_generator()
    discriminator = build_discriminator()

    # Compilação do Discriminador
    discriminator.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

    # GAN
    z = layers.Input(shape=(100,))
    generated_sequence = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_sequence)

    gan = models.Model(z, validity)
    gan.compile(optimizer='adam', loss=BinaryCrossentropy())
    
    return generator, discriminator, gan

def train_gan(generator, discriminator, gan, X, epochs=1, batch_size=32):
    for epoch in range(epochs):
        # Treinar Discriminador
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_sequences = X[idx]
        fake_sequences = generator.predict(np.random.randn(batch_size, 100))

        d_loss_real = discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_sequences, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Treinar Gerador
        g_loss = gan.train_on_batch(np.random.randn(batch_size, 100), np.ones((batch_size, 1)))
        
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

def generate_music(generator, duration, seq_length):
    noise = np.random.randn(duration * seq_length, 100)
    generated_sequences = generator.predict(noise)
    return generated_sequences

class MusicGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MusicaGen - Music Generator")
        
        self.label = tk.Label(root, text="Selecione músicas para treinamento:")
        self.label.pack(pady=10)
        
        self.select_button = tk.Button(root, text="Selecionar Músicas", command=self.select_files)
        self.select_button.pack(pady=5)
        
        self.train_button = tk.Button(root, text="Treinar Modelo", command=self.train_model)
        self.train_button.pack(pady=5)
        
        self.generate_button = tk.Button(root, text="Gerar Música", command=self.generate_music)
        self.generate_button.pack(pady=5)
        
        self.generate_midi_button = tk.Button(root, text="Gerar MIDI", command=self.generate_midi)
        self.generate_midi_button.pack(pady=5)
        
        self.duration_label = tk.Label(root, text="Duração da Música (segundos):")
        self.duration_label.pack(pady=5)
        
        self.duration_entry = tk.Entry(root)
        self.duration_entry.pack(pady=5)
        
        self.selected_files = []
        self.generator = None
        self.discriminator = None
        self.gan = None
        
    def select_files(self):
        files = filedialog.askopenfilenames(title="Selecione Músicas", filetypes=[("Arquivos de Áudio", "*.mp3 *.wav")])
        self.selected_files = self.root.tk.splitlist(files)
        self.label.config(text=f"Arquivos Selecionados: {len(self.selected_files)}")
        
    def train_model(self):
        if not self.selected_files:
            self.label.config(text="Nenhum arquivo selecionado!")
            return
        
        self.label.config(text="Carregando e processando dados...")
        data = load_and_preprocess_audio(self.selected_files)
        seq_length = 30
        X = create_sequences(data, seq_length)
        
        self.label.config(text="Construindo e treinando modelo...")
        input_shape = (X.shape[1], X.shape[2])
        self.generator, self.discriminator, self.gan = build_gan(input_shape)
        train_gan(self.generator, self.discriminator, self.gan, X, epochs=1, batch_size=32)
        
        self.label.config(text="Modelo treinado com sucesso!")
    
    def generate_music(self):
        if self.generator is None:
            self.label.config(text="Modelo não treinado!")
            return
        
        try:
            duration = int(self.duration_entry.get())
        except ValueError:
            self.label.config(text="Por favor, insira uma duração válida!")
            return
        
        self.label.config(text="Gerando música...")
        generated_sequences = generate_music(self.generator, duration, 30)
        generated_spectrogram = np.vstack(generated_sequences)
        
        self.label.config(text="Convertendo espectrograma para áudio...")
        generated_audio = librosa.feature.inverse.mel_to_audio(generated_spectrogram.T, sr=22050, n_iter=512)
        
        wav_path = 'musica_gerada.wav'
        sf.write(wav_path, generated_audio, 22050)
        self.label.config(text="Música gerada e salva com sucesso!")

    def generate_midi(self):
        if self.generator is None:
            self.label.config(text="Modelo não treinado!")
            return
        
        try:
            duration = int(self.duration_entry.get())
        except ValueError:
            self.label.config(text="Por favor, insira uma duração válida!")
            return
        
        self.label.config(text="Gerando MIDI...")
        generated_sequences = generate_music(self.generator, duration, 30)
        generated_spectrogram = np.vstack(generated_sequences)
        
        self.label.config(text="Convertendo espectrograma para MIDI...")
        save_midi(generated_spectrogram, 22050, "musica_gerada_midi")
        self.label.config(text="MIDI gerado e salvo com sucesso!")

root = tk.Tk()
app = MusicGenApp(root)
root.mainloop()
