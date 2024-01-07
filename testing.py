from src.audio.device_settings import prompt_device_selection
from src.audio.record import record_model_data
from src.audio.make_spectrograms import generate_spectrogram
# from src import *

if __name__ == "__main__":

    ###################### TESTING AUDIO ######################

    # audio/device_selection.py

    device, samplerate = prompt_device_selection()
    print(device, samplerate)

    # audio/record.py

    my_recordings = record_model_data(device)
    print(my_recordings)

    # audio/make_spectrograms.py

    import matplotlib.pyplot as plt

    my_spectrogram = generate_spectrogram(
        noise_sample=my_recordings['t'][0], samplerate=samplerate)
    plt.figure(figsize=(2, 2))
    plt.imshow(my_spectrogram)
    plt.show()
