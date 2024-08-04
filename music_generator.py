import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from pydub import AudioSegment
import subprocess
from scipy.signal import butter, lfilter
from midiutil import MIDIFile

# Verificação da Configuração do FFmpeg
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

ffmpeg_installed = check_ffmpeg()

# Funções de Pré-processamento
def load_and_preprocess_audio(file_paths):
    data = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        harmonic, percussive = librosa.effects.hpss(y)
        melody = librosa.feature.rms(y=harmonic)
        
        # Ajustar a dimensão para que possam ser concatenados
        min_length = min(S_dB.shape[1], harmonic.shape[0], percussive.shape[0], melody.shape[1])
        
        S_dB = S_dB[:, :min_length]
        harmonic = harmonic[:min_length]
        percussive = percussive[:min_length]
        melody = melody[:, :min_length]
        
        combined_data = np.vstack((S_dB, harmonic[np.newaxis, :], percussive[np.newaxis, :], melody))
        data.append(combined_data.T)
    return data

def create_sequences(data, seq_length):
    X = []
    y = []
    for song in data:
        for i in range(len(song) - seq_length):
            X.append(song[i:i + seq_length])
            y.append(song[i + seq_length])
    X = np.array(X)
    y = np.array(y)

    # Verifica se y tem a dimensão correta
    if y.ndim == 1:
        y = np.expand_dims(y, axis=-1)

    return X, y

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss=custom_loss, optimizer='adam')
    return model

def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    l2_regularization = 0.01 * tf.reduce_sum(tf.square(y_pred))
    return mse + l2_regularization

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Função para aplicar filtro passa-baixa
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Função para gerar áudio a partir de espectrogramas mel
def spectrogram_to_audio(S_dB, sr=22050):
    S = librosa.db_to_power(S_dB.T)
    audio = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_iter=512)
    return audio

# Função para salvar o áudio em .wav e .mp3
def save_audio(audio, sr, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filtered_audio = lowpass_filter(audio, cutoff=8000, fs=sr, order=6)
    
    wav_path = os.path.join('data', filename + '.wav')
    mp3_path = os.path.join('data', filename + '.mp3')
    
    sf.write(wav_path, filtered_audio, sr)
    
    if ffmpeg_installed:
        try:
            sound = AudioSegment.from_wav(wav_path)
            sound.export(mp3_path, format='mp3')
        except Exception as e:
            print(f"Erro ao converter para MP3: {e}")
    else:
        print("FFmpeg não está instalado ou não está configurado corretamente. Instale o FFmpeg para salvar em MP3.")

# Função para salvar o espectrograma como arquivo MIDI
def save_midi(spectrogram, sr, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)  # Adiciona tempo ao MIDI (exemplo: 120 BPM)
    
    for i, frame in enumerate(spectrogram):
        for j, amplitude in enumerate(frame):
            if amplitude > -20:  # Threshold para adicionar notas (ajustável)
                midi.addNote(0, 0, j, i / sr * 512, 0.5, int(amplitude))
    
    midi_path = os.path.join('data', filename + '.mid')
    with open(midi_path, "wb") as output_file:
        midi.writeFile(output_file)

# Classe Tkinter para Interface Gráfica
class MusicGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MusicaGen - Music Generator")
        
        self.label = tk.Label(root, text="Selecione músicas para treinamento:")
        self.label.pack(pady=10)
        
        self.select_button = tk.Button(root, text="Selecionar Músicas", command=self.select_files)
        self.select_button.pack(pady=5)
        
        self.train_button = tk.Button(root, text="Treinar Modelo", command=self.train_model, state=tk.DISABLED)
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
        self.sr = 22050  # Taxa de amostragem padrão para áudio
        
        self.music_listbox = tk.Listbox(root, height=10, width=50)
        self.music_listbox.pack(pady=10)
        
        self.input_shape_dim = None
    
    def select_files(self):
        files = filedialog.askopenfilenames(title="Selecione Músicas", filetypes=[("Arquivos de Áudio", "*.mp3 *.wav")])
        self.selected_files = self.root.tk.splitlist(files)
        self.music_listbox.delete(0, tk.END)
        for file in self.selected_files:
            self.music_listbox.insert(tk.END, file)
        self.label.config(text=f"Arquivos Selecionados: {len(self.selected_files)}")
        
        if len(self.selected_files) >= 10:
            self.train_button.config(state=tk.NORMAL)
        else:
            self.train_button.config(state=tk.DISABLED)
    
    def train_model(self):
        if not self.selected_files:
            self.label.config(text="Nenhum arquivo selecionado!")
            return
        
        self.label.config(text="Carregando e processando dados...")
        data = load_and_preprocess_audio(self.selected_files)
        seq_length = 30  # Comprimento da sequência de entrada
        X, y = create_sequences(data, seq_length)
        
        self.label.config(text="Construindo e treinando modelo...")
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1]  # Corrige a dimensão da saída
        self.model = build_model(input_shape, output_shape)
        self.model = train_model(self.model, X, y)
        
        self.input_shape_dim = input_shape[1]  # Salva a dimensão do input shape para uso posterior
        
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
        num_samples = duration * self.sr
        generated_audio = np.zeros((num_samples,))
        
        # Inicia com uma sequência aleatória do treinamento
        start_idx = np.random.randint(0, len(self.selected_files))
        initial_audio = load_and_preprocess_audio([self.selected_files[start_idx]])[0]
        generated_sequence = initial_audio[:30]  # Usa o comprimento da sequência
        
        for i in range(0, num_samples, 512):
            prediction = self.model.predict(generated_sequence[np.newaxis, :, :])
            audio_sample = spectrogram_to_audio(prediction[0])
            generated_audio[i:i + len(audio_sample)] = audio_sample[:num_samples - i]
            
            generated_sequence = np.roll(generated_sequence, -1, axis=0)
            generated_sequence[-1, :] = prediction[0]
        
        self.label.config(text="Música gerada com sucesso!")
        save_audio(generated_audio, self.sr, "generated_music")

    def generate_midi(self):
        if self.model is None:
            self.label.config(text="Modelo não treinado!")
            return
        
        self.label.config(text="Gerando MIDI...")
        duration = int(self.duration_entry.get())
        num_samples = duration * self.sr
        generated_spectrogram = np.zeros((num_samples, self.input_shape_dim))
        
        # Inicia com uma sequência aleatória do treinamento
        start_idx = np.random.randint(0, len(self.selected_files))
        initial_audio = load_and_preprocess_audio([self.selected_files[start_idx]])[0]
        generated_sequence = initial_audio[:30]  # Usa o comprimento da sequência
        
        for i in range(num_samples):
            prediction = self.model.predict(generated_sequence[np.newaxis, :, :])
            generated_spectrogram[i] = prediction[0]
            
            generated_sequence = np.roll(generated_sequence, -1, axis=0)
            generated_sequence[-1, :] = prediction[0]
        
        save_midi(generated_spectrogram, self.sr, "generated_music_midi")
        self.label.config(text="MIDI gerado com sucesso!")

# Criação da Interface Gráfica
if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGenApp(root)
    root.mainloop()
