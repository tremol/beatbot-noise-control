# --------- Pretend we're in the root directory ---------
import sys
import os

ROOT_DIR = '../..'
os.chdir(ROOT_DIR)
sys.path.insert(0, os.getcwd())
# -------------------------------------------------------

from src.main.build_run_beatbot import build_beatbot, run_beatbot
from src.main.listen_and_recognize import listen_recognize_and_respond
from src.model.save_load import save_model, load_model
from src.audio.save_load import load_noise_sample_dict
from src.audio.device_settings import prompt_device_selection, get_samplerate
from src.response.print_noise import print_noise
from src.response.keyboard_control import press_key

if __name__ == "__main__":

    ###################### TESTING MODELS ######################

    device = 0  # alternatively, use prompt_device_selection()
    samplerate = get_samplerate(device)

    # model/listen_and_recognize.py

    my_model = load_model(filename='my_test_recordings')
    listen_recognize_and_respond(my_model, print_noise, device=0, duration=5)

    # model/build_run_beatbot.py

    my_recordings = load_noise_sample_dict('my_test_recordings.npy')
    my_model, my_recordings = build_beatbot(device,
                                            starting_noise_data=my_recordings,
                                            skip_recording=False,
                                            save_recordings_filename='my_test_recordings',
                                            rewrite_recordings_file=True,
                                            save_model_filename='my_test_model',
                                            rewrite_model_file=True)

    # print the noises heard
    my_model = load_model('my_test_recordings')
    run_beatbot(my_model, print_noise, device, duration=10)

    # # basic keyboard control
    # my_model = load_model('necrodancer_100each_t-k-p-tsk-cluck')
    # run_beatbot(my_model, press_key, device, duration=120)
