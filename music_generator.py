import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
        data.append(S_dB.T)
    return data

def create_sequences(data, seq_length):
    X = []
    y = []
    for song in data:
        for i in range(len(song) - seq_length):
            X.append(song[i:i + seq_length])
            y.append(song[i + seq_length])
    return np.array(X), np.array(y)

# Funções de Construção e Treinamento do Modelo
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='linear'))
    model.compile(loss=custom_loss, optimizer='adam')
    return model

def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # Adiciona um termo de regularização L2
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
        self.sr = 22050  # Taxa de amostragem padrão para áudio
        
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
        seq_length = 30  # Comprimento da sequência de entrada
        X, y = create_sequences(data, seq_length)
        
        self.label.config(text="Construindo e treinando modelo...")
        input_shape = (X.shape[1], X.shape[2])
        self.model = build_model(input_shape)
        self.model = train_model(self.model, X, y)
        
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
        
        frames_per_second = 22050 // 512
        total_frames = duration * frames_per_second
        
        self.label.config(text="Gerando música...")
        initial_seq = np.random.rand(1, 30, 128)
        
        generated_spectrograms = []
        current_seq = initial_seq
        for _ in range(total_frames):
            next_frame = self.model.predict(current_seq)
            generated_spectrograms.append(next_frame[0])
            current_seq = np.concatenate((current_seq[:, 1:, :], next_frame[:, np.newaxis, :]), axis=1)
        
        generated_spectrogram = np.vstack(generated_spectrograms)
        
        self.label.config(text="Convertendo espectrograma para áudio...")
        generated_audio = spectrogram_to_audio(generated_spectrogram, sr=self.sr)
        
        save_audio(generated_audio, self.sr, "musica_gerada")
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
        
        frames_per_second = 22050 // 512
        total_frames = duration * frames_per_second
        
        self.label.config(text="Gerando música...")
        initial_seq = np.random.rand(1, 30, 128)
        
        generated_spectrograms = []
        current_seq = initial_seq
        for _ in range(total_frames):
            next_frame = self.model.predict(current_seq)
            generated_spectrograms.append(next_frame[0])
            current_seq = np.concatenate((current_seq[:, 1:, :], next_frame[:, np.newaxis, :]), axis=1)
        
        generated_spectrogram = np.vstack(generated_spectrograms)
        
        self.label.config(text="Convertendo espectrograma para MIDI...")
        save_midi(generated_spectrogram, sr=self.sr, filename="musica_gerada")
        self.label.config(text="MIDI gerado e salvo com sucesso!")

root = tk.Tk()
app = MusicGenApp(root)
root.mainloop()
