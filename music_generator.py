import os
import glob
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import torch
import torch.nn as nn
from music21 import converter, instrument, note, chord, stream, tempo

# ==========================================
# 1. MULTI-GENRE MIDI & MOTIF DATASETS
# ==========================================
GENRE_DATASETS = {
    "Classical Piano (Bach / Mozart Style)": {
        "bpm": 120,
        "instrument": instrument.Piano,
        "motifs": [
            "C4", "E4", "G4", "C5", "B4", "G4", "E4", "C4",
            "A3", "C4", "E4", "A4", "G4", "E4", "C4", "A3",
            "F3", "A3", "C4", "F4", "E4", "C4", "A3", "F3",
            "G3", "B3", "D4", "G4", "F4", "D4", "B3", "G3",
            "0.4.7", "E4", "G4", "0.4.7", "A4", "G4", "E4", "D4",
            "9.0.4", "C4", "E4", "9.0.4", "F4", "E4", "C4", "B3",
            "5.9.0", "A3", "C4", "5.9.0", "D4", "C4", "A3", "G3",
            "7.11.2", "B3", "D4", "7.11.2", "E4", "D4", "B3", "C4"
        ]
    },
    "Smooth Jazz & Blues Progression": {
        "bpm": 98,
        "instrument": instrument.ElectricPiano,
        "motifs": [
            "D4", "F4", "A4", "C5", "E5", "D5", "C5", "A4",
            "G3", "B3", "D4", "F4", "A4", "G4", "F4", "D4",
            "C4", "E4", "G4", "B4", "D5", "C5", "B4", "G4",
            "A3", "C4", "E4", "G4", "Bb4", "A4", "G4", "E4",
            "2.5.9.0", "F4", "A4", "2.5.9.0", "C5", "A4", "G4", "F4",
            "7.11.2.5", "B3", "D4", "7.11.2.5", "F4", "E4", "D4", "B3",
            "0.4.7.11", "E4", "G4", "0.4.7.11", "B4", "A4", "G4", "E4"
        ]
    },
    "Cinematic Ambient Arpeggios": {
        "bpm": 85,
        "instrument": instrument.Harp,
        "motifs": [
            "E3", "B3", "E4", "G4", "B4", "G4", "E4", "B3",
            "C3", "G3", "C4", "E4", "G4", "E4", "C4", "G3",
            "G3", "D4", "G4", "B4", "D5", "B4", "G4", "D4",
            "D3", "A3", "D4", "F#4", "A4", "F#4", "D4", "A3",
            "4.7.11", "B4", "G4", "4.7.11", "E4", "F#4", "G4", "B4",
            "0.4.7", "G4", "E4", "0.4.7", "C4", "D4", "E4", "G4"
        ]
    }
}

def get_notes(genre_name):
    midi_files = glob.glob("midi_songs/*.mid")
    if midi_files:
        notes = []
        for file in midi_files:
            midi = converter.parse(file)
            try:
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        return notes, 120, instrument.Piano, f"Loaded {len(midi_files)} custom MIDI file(s)"
    else:
        preset = GENRE_DATASETS[genre_name]
        return preset["motifs"] * 6, preset["bpm"], preset["instrument"], f"Loaded '{genre_name}' training corpus"

def prepare_sequences(notes, seq_length=12):
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    int_to_note = {i: n for i, n in enumerate(pitchnames)}

    network_input, network_output = [], []
    for i in range(0, len(notes) - seq_length):
        network_input.append([note_to_int[n] for n in notes[i:i + seq_length]])
        network_output.append(note_to_int[notes[i + seq_length]])

    X = torch.tensor(network_input, dtype=torch.float32).reshape(len(network_input), seq_length, 1) / float(n_vocab)
    y = torch.tensor(network_output, dtype=torch.long)
    return X, y, network_input, n_vocab, int_to_note

# ==========================================
# 2. DEEP LEARNING LSTM ARCHITECTURE
# ==========================================
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        return self.fc2(out)

# ==========================================
# 3. GUI APPLICATION & AUDIO EXPORT
# ==========================================
class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CodeAlpha - AI Music Generation Studio (PyTorch LSTM + music21)")
        self.root.geometry("700x610")
        self.root.configure(bg="#1e1e2e")
        self.output_file = "ai_generated_music.mid"

        # Header
        header = tk.Frame(root, bg="#181825", pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text="🎹 CodeAlpha AI Music Generation Studio", font=("Segoe UI", 16, "bold"), bg="#181825", fg="#cba6f7").pack()
        tk.Label(header, text="Deep Learning LSTM Sequence Modeling & MIDI Audio Synthesis", font=("Segoe UI", 9), bg="#181825", fg="#a6adc8").pack()

        # Controls Panel
        ctrl_frame = tk.Frame(root, bg="#181825", padx=15, pady=12)
        ctrl_frame.pack(fill=tk.X, padx=15, pady=12)

        # Row 0: Genre Selection
        tk.Label(ctrl_frame, text="Training Style / Dataset:", font=("Segoe UI", 10, "bold"), bg="#181825", fg="#cdd6f4").grid(row=0, column=0, sticky="w", pady=5)
        self.genre_var = tk.StringVar(value=list(GENRE_DATASETS.keys())[0])
        genre_menu = ttk.Combobox(ctrl_frame, textvariable=self.genre_var, values=list(GENRE_DATASETS.keys()), state="readonly", width=36)
        genre_menu.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Row 1: Training Epochs
        tk.Label(ctrl_frame, text="Training Epochs:", font=("Segoe UI", 10, "bold"), bg="#181825", fg="#cdd6f4").grid(row=1, column=0, sticky="w", pady=5)
        self.epochs_var = tk.IntVar(value=60)
        tk.Scale(ctrl_frame, from_=20, to=150, orient=tk.HORIZONTAL, variable=self.epochs_var, bg="#181825", fg="white", highlightthickness=0, length=240).grid(row=1, column=1, padx=10, sticky="w")

        # Row 2: Creativity (Temperature)
        tk.Label(ctrl_frame, text="AI Creativity (Temperature):", font=("Segoe UI", 10, "bold"), bg="#181825", fg="#cdd6f4").grid(row=2, column=0, sticky="w", pady=5)
        self.temp_var = tk.DoubleVar(value=0.85)
        tk.Scale(ctrl_frame, from_=0.4, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, variable=self.temp_var, bg="#181825", fg="white", highlightthickness=0, length=240).grid(row=2, column=1, padx=10, sticky="w")

        # Row 3: Output Length
        tk.Label(ctrl_frame, text="Generated Notes Length:", font=("Segoe UI", 10, "bold"), bg="#181825", fg="#cdd6f4").grid(row=3, column=0, sticky="w", pady=5)
        self.length_var = tk.IntVar(value=64)
        tk.Scale(ctrl_frame, from_=32, to=128, resolution=8, orient=tk.HORIZONTAL, variable=self.length_var, bg="#181825", fg="white", highlightthickness=0, length=240).grid(row=3, column=1, padx=10, sticky="w")

        # Action Buttons
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(pady=5)

        self.train_btn = tk.Button(
            btn_frame, text="⚡ Train LSTM & Generate MIDI", command=self.start_generation,
            bg="#a6e3a1", fg="#11111b", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=16, pady=8, cursor="hand2"
        )
        self.train_btn.pack(side=tk.LEFT, padx=8)

        self.play_btn = tk.Button(
            btn_frame, text="▶ Play Generated Audio", command=self.play_midi, state=tk.DISABLED,
            bg="#89b4fa", fg="#11111b", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=16, pady=8, cursor="hand2"
        )
        self.play_btn.pack(side=tk.LEFT, padx=8)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=660, mode="determinate")
        self.progress.pack(pady=8)

        # Live Console & Note Visualizer
        tk.Label(root, text="Neural Network Training Logs & Generated Melody Sequence:", font=("Segoe UI", 9, "bold"), bg="#1e1e2e", fg="#a6adc8").pack(anchor="w", padx=18)
        self.log_box = scrolledtext.ScrolledText(root, width=80, height=13, font=("Consolas", 9), bg="#11111b", fg="#cdd6f4", relief=tk.FLAT, padx=10, pady=8)
        self.log_box.pack(padx=15, pady=6, fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.root.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def start_generation(self):
        self.train_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.log_box.delete("1.0", tk.END)
        self.progress["value"] = 0
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        try:
            genre = self.genre_var.get()
            epochs = self.epochs_var.get()
            temperature = self.temp_var.get()
            gen_length = self.length_var.get()

            notes, bpm, inst_cls, source_msg = get_notes(genre)
            self.log(f"[DATA] {source_msg} ({len(notes)} tokens)")

            X, y, network_input, n_vocab, int_to_note = prepare_sequences(notes)
            self.log(f"[PREP] Vocabulary Size: {n_vocab} unique pitches/chords | Training Sequences: {len(X)}")

            model = MusicLSTM(input_size=1, hidden_size=128, num_layers=2, output_size=n_vocab)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            self.log(f"[TRAIN] Training 2-Layer LSTM Neural Network ({epochs} epochs)...")
            model.train()
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                pct = (epoch / epochs) * 85
                self.root.after(0, lambda v=pct: self.progress.configure(value=v))
                if epoch % 10 == 0 or epoch == 1:
                    self.log(f"   Epoch [{epoch:03d}/{epochs:03d}] -> Cross-Entropy Loss: {loss.item():.4f}")

            # Generate Music Sequence
            self.log(f"\n[GENERATE] Composing {gen_length} new notes (Temperature = {temperature})...")
            model.eval()
            start_idx = np.random.randint(0, len(network_input) - 1)
            pattern = list(network_input[start_idx])
            generated_notes = []

            with torch.no_grad():
                for _ in range(gen_length):
                    seq_in = torch.tensor(pattern, dtype=torch.float32).reshape(1, len(pattern), 1) / float(n_vocab)
                    pred = model(seq_in)
                    probs = torch.softmax(pred / temperature, dim=-1).numpy().flatten()
                    idx = np.random.choice(len(probs), p=probs)
                    generated_notes.append(int_to_note[idx])
                    pattern = pattern[1:] + [idx]

            self.log("[NOTES] " + " ➔ ".join(generated_notes[:24]) + " ...")

            # Convert to MIDI Stream
            midi_stream = stream.Stream()
            midi_stream.append(tempo.MetronomeMark(number=bpm))
            offset = 0.0
            for item in generated_notes:
                if '.' in item or item.isdigit():
                    chord_notes = []
                    for n in item.split('.'):
                        new_n = note.Note(int(n))
                        new_n.storedInstrument = inst_cls()
                        chord_notes.append(new_n)
                    new_c = chord.Chord(chord_notes)
                    new_c.offset = offset
                    midi_stream.append(new_c)
                else:
                    new_n = note.Note(item)
                    new_n.offset = offset
                    new_n.storedInstrument = inst_cls()
                    midi_stream.append(new_n)
                offset += 0.5

            midi_stream.write('midi', fp=self.output_file)
            self.root.after(0, lambda: self.progress.configure(value=100))
            self.log(f"\n[DONE] Saved playable MIDI track: {os.path.abspath(self.output_file)}")
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))

        except Exception as e:
            self.log(f"[ERROR] {str(e)}")
        finally:
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

    def play_midi(self):
        if os.path.exists(self.output_file):
            os.startfile(self.output_file)
        else:
            messagebox.showwarning("File Missing", "Please generate a MIDI track first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    root.mainloop()