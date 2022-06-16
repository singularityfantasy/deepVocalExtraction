import librosa
import soundfile
import numpy as np
from pathlib import Path
import torch


nfft = 512
hop_len = 128
SAMP_RATE = 22050
SEG_LEN = 2000

p = Path("./data")
model = torch.load('net_track.pth').cuda().eval()
with torch.no_grad():
    for fname in p.glob("*.mp3"):
        try:
            x, fs = librosa.load(fname, sr=SAMP_RATE)
            # https://librosa.org/doc/latest/auto_examples/plot_vocal_separation.html
            x_mel = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=nfft, hop_length=hop_len)
            st = 0
            res = []
            while st + SEG_LEN + 100 < x_mel.shape[1]:  # st + 300
                x_mel_seg = x_mel[:, st:st + SEG_LEN]
                x_mel_seg_tensor = torch.Tensor(x_mel_seg).unsqueeze(0).cuda()
                out = model(x_mel_seg_tensor)
                out = out.data.cpu().squeeze().numpy()  # ** 3
                y_mel_seg = out * x_mel_seg
                res.append(y_mel_seg)
                st += SEG_LEN
            y_mel = np.concatenate(res, axis=1)
            y_foreground = librosa.feature.inverse.mel_to_audio(y_mel, sr=fs, n_fft=nfft, hop_length=hop_len)
            soundfile.write('output/' + fname.name.split('.')[0] + '_fg.flac', y_foreground, fs)

            print("loaded", fname)
        except:
            print("failed to load", fname)
