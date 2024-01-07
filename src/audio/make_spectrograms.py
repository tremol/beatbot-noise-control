# Audio processing: preparing spectrograms

# A function to create a spectrogram from a noise sample recording.

import torch
import torchaudio.transforms


N_MELS = 28                # PARAMETER: the number of mel filterbanks in each spectrogram


def generate_spectrogram(noise_sample, samplerate, n_mels=N_MELS):
    """ Takes a noise_sample as a flattened numpy.array,
    and returns a mel spectrogram as a 2D torch.tensor """

    # normalize to have unit mean, and compute the spectrogram
    normed_sample = torch.from_numpy(noise_sample) / noise_sample.mean()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=samplerate, n_mels=n_mels)(normed_sample)

    return mel.log2()


# # TESTING
# import matplotlib.pyplot as plt
# my_spectrogram = generate_spectrogram( noise_sample=my_recordings['t'][0], , samplerate=samplerate )
# plt.figure(figsize=(2, 2))
# plt.imshow(my_spectrogram)
