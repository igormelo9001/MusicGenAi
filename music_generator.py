import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError

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

def build_vae(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    shape_before_flattening = tf.keras.backend.int_shape(x)
    x = layers.Flatten()(x)
    
    z_mean = layers.Dense(2)(x)
    z_log_var = layers.Dense(2)(x)
    
    z = layers.Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])
    
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    latent_inputs = layers.Input(shape=(2,))
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(latent_inputs)
    x = layers.Reshape(target_shape=shape_before_flattening[1:])(x)
    x = layers.Conv1DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    outputs = layers.Conv1DTranspose(input_shape[1], 3, activation='sigmoid', padding='same')(x)
    
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    
    outputs = decoder(encoder(inputs)[2])
    vae = models.Model(inputs, outputs, name='vae')
    
    reconstruction_loss = MeanSquaredError()(inputs, outputs)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    
    def vae_loss_fn(y_true, y_pred):
        return MeanSquaredError()(y_true, y_pred) + kl_loss
    
    vae.compile(optimizer='adam', loss=vae_loss_fn)
    return vae, encoder, decoder

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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
        self.model = None
        self.encoder = None
        self.decoder = None
        
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
        self.model, self.encoder, self.decoder = build_vae(input_shape)
        self.model.fit(X, X, epochs=1, batch_size=32)
        
        self.label.config(text="Modelo treinado com sucesso!")
    
    def generate_music(self):
        if self.model is None:
            self.label.config(text="Modelo não treinado!")
            return
        
        try:
            duration = int(self.duration_entry.get())
        except ValueError:
            self.label.config(text="Por favor, insira uma duração válida!")
            return
        
        self.label.config(text="Gerando música...")
        generated_sequences = generate_music_from_vae(self.decoder, 2, num_samples=duration * 5)
        generated_spectrogram = np.vstack(generated_sequences)
        
        self.label.config(text="Convertendo espectrograma para áudio...")
        generated_audio = librosa.feature.inverse.mel_to_audio(generated_spectrogram.T, sr=22050, n_iter=512)
        
        wav_path = 'musica_gerada.wav'
        sf.write(wav_path, generated_audio, 22050)
        self.label.config(text="Música gerada e salva com sucesso!")

    def generate_midi(self):
        if self.model is None:
            self.label.config(text="Modelo não treinado!")
            return
        
        try:
            duration = int(self.duration_entry.get())
        except ValueError:
            self.label.config(text="Por favor, insira uma duração válida!")
            return
        
        self.label.config(text="Gerando MIDI...")
        generated_sequences = generate_music_from_vae(self.decoder, 2, num_samples=duration * 5)
        generated_spectrogram = np.vstack(generated_sequences)
        
        self.label.config(text="Convertendo espectrograma para MIDI...")
        save_midi(generated_spectrogram, 22050, "musica_gerada_midi")
        self.label.config(text="MIDI gerado e salvo com sucesso!")

root = tk.Tk()
app = MusicGenApp(root)
root.mainloop()
