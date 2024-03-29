# --------- Pretend we're in the root directory ---------
import sys
import os

ROOT_DIR = '../..'
os.chdir(ROOT_DIR)
sys.path.insert(0, os.getcwd())
# -------------------------------------------------------

from src.audio.device_settings import prompt_device_selection
from src.audio.record import record_model_data
from src.audio.make_spectrograms import generate_spectrogram
from src.audio.save_load import save_noise_samples, load_noise_samples
import time

if __name__ == "__main__":

    ###################### TESTING AUDIO ######################

    # audio/device_selection.py

    device, samplerate = prompt_device_selection()
    print(device, samplerate)

    # audio/record.py

    # label one recorded noise as 't' for spectrogram below
    my_recordings = record_model_data(device)
    print(my_recordings)

    # audio/make_spectrograms.py

    import matplotlib.pyplot as plt

    my_spectrogram = generate_spectrogram(
        noise_sample=my_recordings['t'][0], samplerate=samplerate)
    plt.figure(figsize=(2, 2))
    plt.imshow(my_spectrogram)
    plt.show()

    # audio/save_load.py

    save_noise_samples(my_recordings, filename='my_recordings.npy', rewrite=True)
    time.sleep(0.5)
    my_loaded_file = load_noise_samples(filename='my_recordings.npy')
    print(my_loaded_file)
