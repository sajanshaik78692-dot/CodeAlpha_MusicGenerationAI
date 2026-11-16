\---



\### 3. `CodeAlpha\_MusicGenerationAI/README.md`

```markdown

\# 🎹 AI Music Generation Studio (PyTorch LSTM + music21)



\*\*CodeAlpha Artificial Intelligence Internship — Task 3\*\*  

\*\*Developer:\*\* Shaik Shajin (`CA/DF1/438274`)



\## 📌 Overview

A deep learning music composition studio that parses musical notes and chords using `music21`, trains a 2-layer PyTorch LSTM neural network on harmonic sequences, and generates original playable `.mid` audio files.



\## ✨ Features

\- \*\*Symbolic MIDI Processing:\*\* Extracts note pitches and polyphonic chord structures using `music21` (supports custom `.mid` files in `midi\_songs/` as well as built-in Classical, Jazz, and Cinematic presets).

\- \*\*2-Layer LSTM Neural Network:\*\* Built in PyTorch to model sequential musical dependencies.

\- \*\*Interactive Studio Controls:\*\* Customize training epochs, sampling creativity (temperature), and output note length with live training loss logs and one-click MIDI playback.



\## 🛠️ Tech Stack

\- \*\*Language:\*\* Python 3

\- \*\*Deep Learning:\*\* `torch` (PyTorch LSTM)

\- \*\*Music Processing:\*\* `music21`, `numpy`

\- \*\*GUI:\*\* `tkinter`



\## 🚀 Installation \& Usage

```bash

git clone \[https://github.com/sajanshaik78692-dot/CodeAlpha\_MusicGenerationAI.git](https://github.com/sajanshaik78692-dot/CodeAlpha\_MusicGenerationAI.git)

cd CodeAlpha\_MusicGenerationAI

pip install torch music21 numpy

python music\_generator.py

