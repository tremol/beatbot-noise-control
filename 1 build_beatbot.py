# Build and run a BeatBot here:
# First select your microphone.
device = 0  # select the microphone. Use sd.query_devices() to see options
print(sd.query_devices())
# Record noise samples and train a model. (~100 samples per noise gives reasonable results.) Save the resulting audio data and model, if you wish:
my_model, my_recordings = build_beatbot(device=device)
# Try out your new model:
# This will run for 10 sec, printing out any noise it hears during that time.
run_beatbot(my_model, print_noise, device=device, duration=10)
# If you already have audio data or a model saved, you can load them with ...
# my_recordings = load_noise_sample_dict(filename='my_noise_data.npy')
# my_model = load_model(filename='my_model.pth')
# You can also build on an old set of recordings with ...
# my_new_model, my_new_recordings = build_beatbot(device=device, starting_noise_data=my_recordings)
# Or skip recording entirely (and testing, if you want) with
# my_new_model, my_new_recordings = build_beatbot(device=device, starting_noise_data=my_recordings,
#                                                 skip_recording=True, skip_testing_model=True)
