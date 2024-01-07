from src.audio.device_settings import prompt_device_selection
from src.audio.record import record_model_data
from src.audio.make_spectrograms import generate_spectrogram
from src.audio.save_load import save_noise_sample_dict, load_noise_sample_dict
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

    save_noise_sample_dict(my_recordings)  # enter filename: my_recordings
    time.sleep(0.5)
    my_loaded_file = load_noise_sample_dict(filename='my_recordings.npy')
    print(my_loaded_file)
