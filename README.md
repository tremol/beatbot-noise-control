# BeatBot Noise Control

The goal of this project is to build an easy-to-use and self-contained system to record yourself making percussive noises (think: beatboxing), teach your computer to recognize them, then have your computer listen for them and respond.

It can detect, learn to recognize, and respond to percussive noises (like short consonant sounds), but not longer noises (like a hisssss). My goal was just to make something that works, but even in its current state just a couple minutes of audio recording is enough to train a model decent enough to play a simple game, like *[Crypt of the Necrodancer](https://store.steampowered.com/app/247080/Crypt_of_the_NecroDancer/)*.

This was a ~week-long learning project to explore machine learning, inspired by other [hands-free computing projects](#acknowledgments).

## Getting Started

- Run `beatbot_demo.py`. It will walk you through recording, training, and trying out your model.

- For a usage example, check out `beatbot_necrodancer.py`, which responds to noises by pressing up/down/left/right/escape keys.
    - Note: It may not work well for you, since it was trained by me on my microphone, but hopefully the code will be illustrative. (The [tsk](https://www.youtube.com/watch?v=2BsMktG9ruw&t=16s) and [cluck](https://www.youtube.com/watch?v=4MiKHpcvI9M&t=65s) are like these.)

### Example Setup

Here are two ways to set this up with a Unix/macOS terminal. I recommend using local virtual environments for good hygiene.

Clone/download this repo. Once inside ...

Use pip to install requirements. (If you don't want to use a virtual environment, just skip the first two lines.)
```
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> python beatbot_demo.py
```

Alternatively, use Conda:
```
> conda env create --prefix ./.env -f environment.yml
> conda activate ./.env
> python beatbot_demo.py
```

### Recording Suggestions

I suggest short consonant sounds like "t", "p", or "k", or other clicks or pops you can make with your lips or tongue. Quick combinations like "ps" or "tf" can also work well. 

I find it only takes ~50-100 samples to get a decent (not excellent) model, and it takes less than a minute to record 100 samples of a noise. So you can experiment with different noises and record lots of data for training different models quickly. Recognition quality also depends on the quality of your microphone, of course.

Note: The model will return a prediction for any percussive noise it hears. There is no capability to reject unfamiliar noises, so be aware of this when attaching keyboard simulation to noise recognition. A microphone with better noise rejection can help to reduce false positives, but regardless you may want to use headphones if using this while playing music or a game.

## Code structure

- `explorations (development)` - jupyter notebooks and audio files used for [development](#development-process-for-the-curious); all are non-essential now
- `output` - saved recordings and models
- `src` - the primary code, adapted from the `BeatBot - all together` development notebook
    - `audio` - audio listening, recording, and processing
    - `model` - neural network definition, training, and evaluation
    - `main` - bringing audio and model code together to build and use models
    - `response` - functions to respond to recognized noises, e.g., by pressing keys
    - `utils` - misc generic functions
    - `beatbot_demo.py` and `beatbot_necrodancer.py` - example usage, see [Getting Started](#getting-started)


### Key dependencies

[Setup](#example-setup) will install the following key dependencies:
* `matplotlib` - for visualizing model quality (via the confusion matrix); also visualizing spectrograms in testing
* `numpy` - various processing tasks
* `PyAutoGUI` - keyboard control
* `scikit-learn` - computing the confusion matrix
* `sounddevice` - getting audio input
* `torch` - machine learning
* `torchaudio` - making spectrograms to train models on

Optionally, you may need `ipykernel` to view the Jupyter notebooks. If you don't already have Jupyter, here is the [Jupyter installation guide](https://jupyter.readthedocs.io/en/latest/install.html).

## Development process, for the curious

The jupyter notebooks (Explorations 1-3) chronicle the learning and development process behind the code. It originally came together in the "BeatBot - all together" notebook, and that's where the project stopped — you just ran that notebook. Returning to the project in early 2024, however, it no longer worked. So I extracted and updated the code to its current form.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was inspired by the hands-free computing projects

* [Talon](https://talonvoice.com)
* [PopClick](https://github.com/trishume/PopClick)
* [Parrot.PY](https://github.com/chaosparrot/parrot.py)

In fact, most of this project was written hands-free using Talon.

Talon also has a robust expansion of its own noise recognition system in the works. If you'd like to contribute to that effort, you can record noise samples at [noise.talonvoice.com](https://noise.talonvoice.com). I am not a developer on that project, but am excited to see it advance.
