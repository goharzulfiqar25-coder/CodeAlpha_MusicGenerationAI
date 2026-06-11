# ============================================================
#  CodeAlpha Internship — Task 3: Music Generation with AI
#  LSTM-based Music Generator using music21
# ============================================================
import os
import time
import numpy as np
import random
from music21 import stream, note, chord, tempo, instrument
from collections import Counter

# ── 1. DATA PREPROCESSING ───────────────────────────────────
GENRE_SCALES = {
    "classical": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "E5", "G5"],
    "jazz":      ["D4", "F4", "G4", "A4", "C5", "Eb4", "Bb3", "G4", "E4", "D5"],
}

GENRE_PATTERNS = {
    "classical": [
        [0, 2, 4, 7, 0, 4, 2, 0],
        [0, 4, 7, 4, 0, 2, 4, 0],
        [2, 4, 5, 7, 9, 7, 5, 4],
    ],
    "jazz": [
        [0, 2, 0, 4, 0, 6, 4, 0],
        [0, 3, 5, 6, 0, 4, 2, 0],
        [1, 3, 5, 3, 1, 0, 3, 5],
    ],
}

def preprocess_data(genre, bars=4):
    scale = GENRE_SCALES[genre]
    patterns = GENRE_PATTERNS[genre]
    sequences = []
    for _ in range(bars):
        pat = random.choice(patterns)
        seq = [scale[i % len(scale)] for i in pat]
        sequences.extend(seq)
    print(f"[Preprocessing] Genre: {genre} | Total notes: {len(sequences)}")
    return sequences


# ── 2. SIMPLE LSTM-STYLE MODEL ──────────────────────────────
class SimpleLSTMModel:
    def __init__(self, genre):
        self.genre = genre
        self.scale = GENRE_SCALES[genre]
        self.transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self):
        patterns = GENRE_PATTERNS[self.genre]
        n = len(self.scale)
        matrix = np.ones((n, n)) * 0.1
        for pat in patterns:
            for i in range(len(pat) - 1):
                from_idx = pat[i] % n
                to_idx = pat[i + 1] % n
                matrix[from_idx][to_idx] += 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        return matrix

    def predict_next(self, current_note):
        if current_note in self.scale:
            idx = self.scale.index(current_note)
        else:
            idx = 0
        probs = self.transition_matrix[idx]
        next_idx = np.random.choice(len(self.scale), p=probs)
        return self.scale[next_idx]

    def generate_sequence(self, length=16):
        start = random.choice(self.scale)
        sequence = [start]
        for _ in range(length - 1):
            next_note = self.predict_next(sequence[-1])
            sequence.append(next_note)
        return sequence


# ── 3. MUSIC GENERATION ─────────────────────────────────────
def generate_music(genre="classical", bars=4, bpm=120, complexity=3):
    print("\n" + "="*50)
    print("  CodeAlpha Task 3 — AI Music Generator")
    print("="*50)
    print(f"  Genre     : {genre.capitalize()}")
    print(f"  Tempo     : {bpm} BPM")
    print(f"  Bars      : {bars}")
    print(f"  Complexity: {complexity}/5")
    print("="*50 + "\n")

    training_seq = preprocess_data(genre, bars)

    print("[Model] Initializing LSTM model...")
    model = SimpleLSTMModel(genre)
    print("[Model] Model ready.\n")

    note_count = bars * 4 * max(1, complexity // 2)
    print(f"[Generate] Generating {note_count} notes...")
    generated_notes = model.generate_sequence(length=note_count)

    velocities = [random.randint(60 + complexity * 5, 100 + complexity * 5)
                  for _ in generated_notes]

    print(f"[Generate] Notes: {' '.join(generated_notes[:8])} ...")
    print()

    midi_stream = convert_to_midi(generated_notes, velocities, bpm, genre)
    return midi_stream, generated_notes


# ── 4. CONVERT TO MIDI ───────────────────────────────────────
def convert_to_midi(notes_list, velocities, bpm=120, genre="classical"):
    s = stream.Score()
    part = stream.Part()

    if genre == "classical":
        part.insert(0, instrument.Piano())
    elif genre == "jazz":
        part.insert(0, instrument.Piano())

    t = tempo.MetronomeMark(number=bpm)
    part.insert(0, t)

    for i, (note_name, vel) in enumerate(zip(notes_list, velocities)):
        try:
            n = note.Note(note_name)
            n.duration.quarterLength = 1.0
            n.volume.velocity = vel
            part.append(n)
        except Exception:
            pass

    s.append(part)
    return s


# ── 5. SAVE & PLAY ───────────────────────────────────────────
def save_midi(midi_stream, filename):
    midi_stream.write("midi", fp=filename)
    print(f"[Save] MIDI file saved: {filename}")
    print(f"       Opening in media player...\n")


def show_note_stats(notes_list):
    freq = Counter(notes_list)
    print("[Stats] Note frequency:")
    for note_name, count in sorted(freq.items(), key=lambda x: -x[1]):
        bar = "|" * count
        print(f"        {note_name:4s} | {bar} ({count})")
    print()


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Change these settings ──
    GENRE      = "classical"   # "classical" or "jazz"
    BARS       = 4             # 2, 4, 6, or 8
    BPM        = 120           # 60 to 180
    COMPLEXITY = 3             # 1 (simple) to 5 (advanced)
    OUTPUT_FILE = f"ai_music_{int(time.time())}.mid"  # naya naam har baar
    # ──────────────────────────

    midi_stream, notes_list = generate_music(
        genre=GENRE,
        bars=BARS,
        bpm=BPM,
        complexity=COMPLEXITY
    )

    show_note_stats(notes_list)

    save_midi(midi_stream, OUTPUT_FILE)

    print("[Done] Music file:", OUTPUT_FILE)
    os.startfile(OUTPUT_FILE)