# From recording to recognizing: all together
# Here we define the two main functions:
# * `build_beatbot` to record training audio + train the model + evaluate the model, and
# * `run_beatbot` to continuously listen + recognize noises + act on them.

from src.main.listen_and_recognize import listen_recognize_and_respond
from src.audio.device_settings import get_samplerate
from src.audio.record import record_model_data
from src.model.prepare_datasets import NoisesDataset, prepare_even_data_loaders
from src.model.define_model import Net
from src.model.train_model import train_net
from src.model.evaluate_model import accuracy_rating, plot_confusion_matrix
from src.audio.save_load import save_noise_samples
from src.model.save_load import save_model


def run_beatbot(model, act_on_noise, device, duration):
    noises = ', '.join(list(model.noise_int_to_str.values()))
    print(f'This model recognizes the noises: {noises}')
    
    listen_recognize_and_respond(model, act_on_noise, device, duration)

def build_beatbot(device, starting_noise_data={},
                  skip_recording=False,
                  batch_size=8, epochs=10, batch_progress=100,
                  skip_testing_model=False,
                  save_recordings_filename=None, rewrite_recordings_file=False,
                  save_model_filename=None,      rewrite_model_file=False):
    """ Record audio data, train a model, evaluate it, and optionally save the results.
    If skip_testing_model is True, use all data for training and skip the model testing."""

    # Record training data and construct the dataset
    if skip_recording:
        noise_data_dict = starting_noise_data
    else:
        noise_data_dict = record_model_data(
            device=device, starting_noise_data=starting_noise_data)
    
    print('Building model from recordings ...')

    samplerate = get_samplerate(device)

    dataset = NoisesDataset(noise_data_dict, samplerate)

    # Prepare the dataloader. Use all data as training data if skip_testing_model is True.
    if skip_testing_model:
        training_fraction = 1
    else:
        training_fraction = 0.8
    train_loader, test_loader, _, _ = prepare_even_data_loaders(dataset, samplerate, batch_size=batch_size,
                                                                training_fraction=training_fraction)

    # Build and train the neural net
    image_size = dataset[0][0].size()
    model = Net(image_size, dataset.noise_int_to_str)
    train_net(model, epochs, train_loader, batch_progress=batch_progress)

    # Evaluate the model and show the confusion matrix, or skip testing altogether.
    if not skip_testing_model:
        preds, targets = accuracy_rating(model, test_loader, 'test')
        plot_confusion_matrix(preds, targets, dataset.noise_int_to_str)

    # Offer to save the recordings and model
    print('\nWould you like to save your audio data?')
    save_noise_samples(
        noise_data_dict, filename=save_recordings_filename, rewrite=rewrite_recordings_file)
    print('\nWould you like to save your model?')
    save_model(model, filename=save_model_filename, rewrite=rewrite_model_file)

    return model, noise_data_dict


# # TESTING
# my_model, my_recordings = build_beatbot(device=0, starting_noise_data=my_recordings)
# # TESTING
# listen_recognize_and_respond(my_model, print_noise, device=0, duration=20)
