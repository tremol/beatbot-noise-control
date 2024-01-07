## Real-time listening: general functions
# Functions to continuously listen for noises, and pass them to a processing function.
import sounddevice as sd
import numpy as np
import queue
import time
########### PARAMETERS ###########

device = 0 # select the microphone. Use sd.query_devices() to see options
print(sd.query_devices())

BATCH_DURATION = 0.02      # listen for noises BATCH_DURATION (seconds) at a time
THRESHOLD_MULTIPLIER = 5   # detect a spike when the next batch is at least THRESHOLD_MULTIPLIER times bigger
THRESHOLD_ABSOLUTE = 0.005 # ignore any spikes that don't rise above this. Too many false positives without this
BATCHES_PER_NOISE = 3      # collect BATCHES_PER_NOISE batches of audio input per detected noise

samplerate = sd.query_devices(device, 'input')['default_samplerate']
# optional for future: set the FFT window size based on the sample rate

blocksize = int(samplerate * BATCH_DURATION) # get the block (batch) size in frames
########### Functions for continuous listening and processing ###########

# bundling these is easier than declaring them 'global' in the below
class listen:
    """ Helper variables for processing continuous audio input """
    
    def reset():
        listen.prev_max = 1.
        listen.batches_to_collect = 0
        listen.batches_collected = 0
        listen.current_noise = None
        listen.start = time.time()

        listen.processing_start = 0 # for timing the total processing time
        listen.processing_end = 0

        listen.q_batches = queue.Queue() # a FIFO queue
        listen.all_audio = []  # could use this to collect all audio (uncomment line in callback)
        listen.all_noises = [] # could use this to collect all noises. Use the processing_function to append
    
# The callback function for the sounddevice input stream
def callback(indata, frames, time_pa, status):
    """ Detect if a noise has been made, and add audio to the queue. """
    if status:
        print('STATUS: ', str(status))
    if any(indata):
        indata_copy = indata.copy()
        new_max = np.absolute(indata_copy).max()
        # listen.all_audio.append(indata_copy)
        
        # Gather audio data if more is required. Make sure to *copy* the input data.
        if listen.batches_to_collect > 0:
            listen.q_batches.put_nowait(indata_copy)
            listen.batches_collected  += 1
            listen.batches_to_collect -= 1
                
        # Otherwise, see if a new noise has been detected
        elif ( new_max > THRESHOLD_ABSOLUTE and
               new_max > THRESHOLD_MULTIPLIER * listen.prev_max ):
            
            listen.processing_start = time.time()
            
            listen.q_batches.put_nowait(indata_copy)
            listen.batches_collected += 1
            listen.batches_to_collect = BATCHES_PER_NOISE - 1 # get more batches
               
        listen.prev_max = new_max
        
    else:
        print('no input')

# Returns True if enough time has elapsed
def time_elapsed(duration):
    def _time_elapsed():
        return time.time() - listen.start > duration
    return _time_elapsed

# A helper to print the time it took to process a single noise recognition
def print_processing_time():
    listen.processing_end = time.time()
    print('Processing took {:.4f} sec\n'.format(
        listen.processing_end - listen.processing_start))

# The main generic real-time listening function
def listen_and_process(processing_function, stop_condition=time_elapsed(3),
                       device=device, print_after_processing=None):
    """ Listen continuously for noises until stop_condition() returns True (default: wait 3 sec).
    As each noises heard, process is using processing_function. Return all noises at the end. """

    listen.reset() # reinitialize helper variables
    
    with sd.InputStream(device=device, channels=1, callback=callback,
                        blocksize=blocksize,
                        samplerate=samplerate):
        print('Listening...')
        while True:
            
            # data collects if it meets the threshold. Process when enough data is in queue:
            if listen.batches_collected >= BATCHES_PER_NOISE:
                data = []
                for _ in range(BATCHES_PER_NOISE):
                    data.append( listen.q_batches.get_nowait() )
                listen.batches_collected -= BATCHES_PER_NOISE
                
                listen.current_noise = np.concatenate( data, axis=None )
                
                processing_function( listen.current_noise )
                
                # print something after processing, if desired
                print_after_processing() if print_after_processing else None
                    
            # listen until the condition is met
            if stop_condition():
                break
        print('Done.')