import stempeg
from pathlib import Path
import librosa
import h5py

nfft = 512
hop_len = 128
SAMP_RATE = 22050
SEG_LEN = 10

frame_per_sec = int(SAMP_RATE / hop_len)
mel_seg_len = frame_per_sec * SEG_LEN


def proc_dset(data_path, output_fn):
    p = Path(data_path)
    idx = 0
    file_idx = 1
    with h5py.File(output_fn, "w") as f:
        dset_x = f.create_dataset("feature_x", (1, 128, mel_seg_len), maxshape=(None, 128, mel_seg_len))
        dset_y = f.create_dataset("feature_y", (1, 128, mel_seg_len), maxshape=(None, 128, mel_seg_len))

        for fn in p.glob("*.mp4"):
            try:
                audio, sample_rate = stempeg.read_stems(fn.as_posix(), sample_rate=SAMP_RATE)
                x = audio[0].max(-1)
                y = audio[-1].max(-1)

                x_mel = librosa.feature.melspectrogram(y=x, sr=sample_rate, n_fft=nfft, hop_length=hop_len)
                y_mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=nfft, hop_length=hop_len)

                st_pos = 0
                while st_pos + mel_seg_len < x_mel.shape[1]:
                    x_mel_seg = x_mel[:, st_pos:st_pos + mel_seg_len]
                    y_mel_seg = y_mel[:, st_pos:st_pos + mel_seg_len]

                    if idx > 0:
                        dset_x.resize((idx + 1, 128, mel_seg_len))
                        dset_y.resize((idx + 1, 128, mel_seg_len))
                    dset_x[idx] = x_mel_seg
                    dset_y[idx] = y_mel_seg

                    st_pos += mel_seg_len
                    idx += 1
                print("loaded", file_idx, fn)
                file_idx += 1

            except:
                print("error", fn)


if __name__ == '__main__':
    proc_dset(r"D:\musdb18\test", "test_features.hdf5")
    proc_dset(r"D:\musdb18\train", "train_features.hdf5")
