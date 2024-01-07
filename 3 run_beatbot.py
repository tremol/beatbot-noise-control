# And define a model that only recognizes these five noises, even if we recorded more earlier:
my_recordings_subset = {k: v for k, v in my_recordings.items() if k in (
    't', 'k', 'p', 'tsk', 'cluck')}
necrodancer_model, _ = build_beatbot(
    device=device, skip_recording=True, starting_noise_data=my_recordings_subset)
# Now we make it run for 20 min, switch over to the game, and play!
run_beatbot(necrodancer_model, press_key, device=device, duration=20*60)
