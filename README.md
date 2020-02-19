# BeatBot Noise Control

The goal of this project is to build an accessible and self-contained system to record yourself making noises (think: beatboxing), teach your computer to recognize them, then have your computer listen for them and respond. In its present state, it can detect, learn to recognize, and respond to percussive noises (like short consonant sounds), but not longer noises (like a hisssss). There is much room for improvement, but even in its current state just a couple minutes of audio recording is enough to train a model decent enough to play *Crypt of the Necrodancer*.

## Getting Started

### Just run the BeatBot notebook!

There is currently only one essential file:  `BeatBot - all together.ipynb`. Once you have Jupyter and the requisite Python libraries (see Prerequisites) download that file, open it, and follow instructions at the top. The prompts will guide you through recording audio data to train your own model. Once you have a model, you can use the sample code to simulate keyboard presses from noise recognitions, which is all you need to try it out on a game or other application.

### Recording Suggestions

I suggest short consonant sounds like "t", "p", or "k", or other clicks or pops you can make with your lips or tongue. Quick combinations like "ps" or "tf" can also work well. 

I find it only takes ~100 samples to get a decent (not excellent) model, and it takes less than a minute to record 100 samples of a noise, so you can experiment with different noises and record lots of data for training different models quickly. On the flipside, if you want to use datasets this small you'll want to record new models for different people or microphones. (A game like *Necrodancer* only needs a few buttons, however, so this only takes a few minutes start to finish.)

Note: At present the model will return a prediction for any percussive noise it hears. There is no capability yet to reject unfamiliar noises, so be aware of this when attaching keyboard simulation to noise recognition. A microphone with better noise rejection can help to reduce false positives, but regardless you may want to use headphones if using this while playing music or a game.

### Prerequisites

At present, the project is all in Jupyter notebooks, and written in Python. If you don't already have Jupyter, here is the [Jupyter installation guide](https://jupyter.readthedocs.io/en/latest/install.html).

Once you have Jupyter, you can start it by typing the command `jupyter notebook` in a terminal. It should open in your browser, at which point you can navigate to the BeatBot notebook and open it.

The essential `BeatBot` notebook imports the following Python libraries:

* sounddevice
* numpy
* queue
* time
* IPython
* torch
* torchaudio
* matplotlib
* itertools
* sklearn
* os
* pyautogui

I personally use Anaconda as a Python distribution, and was able to install everything it didn't already include via

`conda install -c conda-forge LIBRARYNAME`

### Development process, for the curious

The other notebooks (Explorations 1-3) chronicle the learning and development process behind the code that comes together in the BeatBot notebook. Check them out if you're interested, but they aren't necessary for the final product.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This project was inspired by the hands-free computing projects

* [Talon](https://talonvoice.com)
* [PopClick](https://github.com/trishume/PopClick)
* [Parrot.PY](https://github.com/chaosparrot/parrot.py)

In fact, most of this project was written hands-free using Talon.

Talon also has a robust expansion of its own noise recognition system in the works. If you'd like to contribute to that effort, you can record noise samples at [noise.talonvoice.com](https://noise.talonvoice.com). I am not a developer on that project, but am excited to see it advance.
