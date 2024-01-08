# Real-time listening and recognition
# With the trained model in hand, make a listener to recognize noises and act on them.


from src.audio.device_settings import get_samplerate
from src.audio.make_spectrograms import N_MELS, generate_spectrogram
from src.audio.listen import listen_and_process, time_elapsed
import torch

# Note this supposes that the samplerate and n_mels for get_prediction are the
# same as those used for the training dataset. This could be made more robust by
# allowing processing_function to accept the sample rate as well as a numpy
# array, and by recording the n_mels used in the dataset the model was trained
# on.

def get_prediction(model, noise_sample, device, n_mels=N_MELS):
    """ Build the spectrogram and use our model to recognize the noise """

    samplerate = get_samplerate(device)
    mel = generate_spectrogram(noise_sample, samplerate, n_mels)

    # change from torch.Size([A, B]) to torch.Size([1, 1, A, B])
    mel = mel[None, None, :, :]

    # run through the model and get prediction
    output = model(mel)
    energy, label = torch.max(output.data, 1)

    # return the string label of the noise
    return model.noise_int_to_str[label.item()]


def listen_recognize_and_respond(model, act_on_noise, device, duration=5):
    """ Continuously listen for noises for duration (sec), then recognize them
    with the model and respond with the function act_on_noise. """

    def processing_function(noise_sample):
        pred = get_prediction(model, noise_sample, device)
        act_on_noise(pred)

    listen_and_process(processing_function=processing_function,
                       stop_condition=time_elapsed(duration),
                       device=device,
                       print_after_processing=None)

# # TESTING
# listen_recognize_and_respond(my_net, print_noise, device=2, duration=20)
