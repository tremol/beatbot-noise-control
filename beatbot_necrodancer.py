from src.audio.device_settings import prompt_device_selection
from src.main.build_run_beatbot import build_beatbot, run_beatbot
from src.audio.save_load import load_noise_sample_dict
from src.model.save_load import load_model
from src.response.keyboard_control import KEYBOARD_MAPPING, press_key

# This runs a specific model for basic keyboard control:
# up/down/left/right/escape. This is enough to play Crypt of the Necrodancer,
# for instance (a roguelike rhythm game, well-suited to hand-free play:
# https://store.steampowered.com/app/247080/Crypt_of_the_NecroDancer/)

if __name__ == "__main__":

    # First select the right microphone. This model was recorded on my Sennheiser.
    device = prompt_device_selection()

    # Load the model.
    necrodancer_model = load_model('necrodancer_100each_t-k-p-tsk-cluck')

    # Now we make it run for 20 min, switch over to the game, and play!
    minutes = 20
    run_beatbot(necrodancer_model, press_key, device, duration=60*minutes)


    ########## Code used to build the model ##########

    # necrodancer_model, _ = build_beatbot(device=device,
    #                                     skip_recording=False,
    #                                     skip_testing_model=True,
    #                                     starting_noise_data={},
    #                                     save_recordings_filename='necrodancer_100each_t-k-p-tsk-cluck',
    #                                     save_model_filename='necrodancer_100each_t-k-p-tsk-cluck',
    #                                     )

    # To use just part of a previous recording set, try using this as starting
    # point. It's better to avoid extraneous noises if you don't want them
    # mapped to keys.

    # my_recordings = load_noise_sample_dict(...)
    # my_recordings_subset = {k: v for k, v in my_recordings.items() if k in
    #                         KEYBOARD_MAPPING.keys()}
