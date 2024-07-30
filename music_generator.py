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
        y, sr = librosa.load(file_path, sr=22050)  # Carrega o arquivo de áudio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extrai MFCCs
        data.append(mfccs.T)  # Transpõe para que a sequência de tempo seja a primeira dimensão
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
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(40, activation='linear'))  # Saída com 40 MFCCs
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Função para gerar áudio a partir de MFCCs
def mfcc_to_audio(mfccs, sr=22050):
    # Converte MFCCs para o inverso do espectrograma de potência
    S = librosa.feature.inverse.mfcc_to_mel(mfccs.T)
    audio = librosa.feature.inverse.mel_to_audio(S, sr=sr)
    return audio

# Função para salvar o áudio em .wav e .mp3
def save_audio(audio, sr, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    
    wav_path = os.path.join('data', filename + '.wav')
    mp3_path = os.path.join('data', filename + '.mp3')
    
    sf.write(wav_path, audio, sr)
    
    if ffmpeg_installed:
        try:
            sound = AudioSegment.from_wav(wav_path)
            sound.export(mp3_path, format='mp3')
        except Exception as e:
            print(f"Erro ao converter para MP3: {e}")
    else:
        print("FFmpeg não está instalado ou não está configurado corretamente. Instale o FFmpeg para salvar em MP3.")

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
        
        frames_per_second = 22050 // 512  # Número de frames por segundo (assumindo hop_length=512)
        total_frames = duration * frames_per_second
        
        self.label.config(text="Gerando música...")
        # Gere uma sequência inicial aleatória
        initial_seq = np.random.rand(1, 30, 40)
        
        generated_mfccs = []
        current_seq = initial_seq
        for _ in range(total_frames):  # Gera frames suficientes para a duração desejada
            next_frame = self.model.predict(current_seq)
            generated_mfccs.append(next_frame[0])
            current_seq = np.append(current_seq[:, 1:, :], next_frame.reshape(1, 1, 40), axis=1)
        
        generated_mfccs = np.array(generated_mfccs)
        
        audio = mfcc_to_audio(generated_mfccs, sr=self.sr)
        save_audio(audio, self.sr, "generated_music")
        
        self.label.config(text=f"Música gerada com sucesso! Arquivos salvos em 'data/generated_music.wav' e 'data/generated_music.mp3'")
    
def main():
    root = tk.Tk()
    app = MusicGenApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
