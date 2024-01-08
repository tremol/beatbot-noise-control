from src.audio.device_settings import prompt_device_selection
from src.audio.save_load import load_noise_samples
from src.main.build_run_beatbot import build_beatbot, run_beatbot
from src.response.print_noise import print_noise

# Build and try out a BeatBot here:

if __name__ == "__main__":

    # First select your microphone.
    device = prompt_device_selection()

    # Start fresh if you don't have any recordings yet.
    # my_recordings = {}

    # Alternatively, start by loading a file to build on previous recordings
    my_recordings = load_noise_samples('demo_recordings.npy')

    # Record noises and train a model. Set skip_testing_model to True to use
    # 100% of samples for training instead of 80%. Try recording just a few
    # (5-20) samples per noise to start. It won't be very accurate, but you can
    # add more later. 50-100 samples (takes under 1 min) per noise works reasonably
    # well.
    my_model, my_recordings = build_beatbot(device,
                                            starting_noise_data=my_recordings,
                                            skip_recording=False,
                                            skip_testing_model=False,
                                            save_recordings_filename='demo_recordings',
                                            rewrite_recordings_file=True,
                                            save_model_filename='demo_model',
                                            rewrite_model_file=True)

    # Try out your new model: This will run for 10 sec, printing out any noise it
    # hears during that time.
    run_beatbot(my_model, print_noise, device, duration=10)
