# master-thesis-code
Code dump from my master thesis on "Fitting Room Acoustic Simulations for Early Reflections to Measured Room Impulse"

Included parts:
- Core algorithm and utilities for extrapolating SDM-SRIRs to different source and listener locations in a room with an image source model (`src` folder, `matlab` folder, see abstract for details on topic)
- Python environment (as created by rye)
- Code used to create main results in thesis, acting as examples for the algorithm (`notebooks` folder)
- webMUSHRA configurations (`webMUSHRA-Customizations` folder)


## Thesis Abstract

**Fitting Room Acoustic Simulations for Early Reflections to Measured Room Impulse** by Johannes Fried

Capturing room acoustics for interactive experiences has applications in cultural heritage, audio production, and marketing. Ideally, a room is captured with few measurements and later heard spatially at any listening position, with arbitrary source position and directivity. This thesis approximates the required changes in early reflections using a simple image source model (ISM) without visibility checking or diffraction. Instead, reflections are extracted from spatial room impulse responses (SRIRs), ignoring invisible reflections and modeling absorption, diffusion, and diffraction by an extracted impulse response. The algorithm builds upon the binaural spatial decomposition method (BinauralSDM) by quantizing the direction of arrival (DoA) to potential image sources instead of arbitrary directions. The ISM is used to extrapolate the SRIR to different source-listener combinations and render binaural room impulse responses (BRIRs), which are evaluated with objective metrics and a listening test.

The main challenge is proper regularization of loudspeaker directivity combined with overlapping reflections, which can cause extracted reflections with much higher energy than the direct sound. It is suggested that a reflection should only be extracted if its energy is less than 6 dB above the direct sound, after compensating for distance and directivity. This strategy greatly improves the perceived quality of extrapolations. The proposed method works slightly better than just extrapolating the direct sound with constant reverb and performs best in cuboid-type rooms with little ISM visibility changes. It might be improved by eliminating spectral distortions and localization errors unrelated to early reflection structure, or by exploring different regularization strategies for loudspeaker directivity.

## Code State

The presented code is one-off research code and only released for posterity. There is no intention of further development.

If you intend to use it, prepare for possible errors on your machine, missing documentation or code, adjusting many things to your use case (lots of stuff is very specialized to my use case) and proceed with caution. Since there is no automated testing, check the output of everything yourself to make sure it works how you expect it to. You can always attempt to open an issue if you have any kind of question and I might respond, though I can't promise anything.

## Requirements

- Git for version control
- [Rye](https://rye.astral.sh/) is used for Python version control. It uses pyproject.toml, so version managers that also use that might work aswell. Note the Python version is tracked in  [.python-version](.python-version)
- MATLAB (tested with R2024a), with requirements of the used [BinauralSDM fork](https://github.com/Firionus/BinauralSDM)
- Jupyter, VSCode, or whatever you use for `.ipynb` notebooks. If you install Juypter, I recommend pipx as version manager.
- For webMUSHRA: Docker

## Installation

- Clone the repo with submodules: 

```console
git clone --recurse-submodules https://github.com/Firionus/master-thesis-code.git
cd master-thesis-code
```

- Download SRIRS: Get the folders `HL05W` and `HL06W` from the [high resolution SRIR dataset](https://doi.org/10.5281/zenodo.10450779) and place them in the folder `data/external/zenodo.10450779`
- Download HRTFs: `Kemar_HRTF_sofa.sofa` from https://doi.org/10.18154/RWTH-2020-11307 and place in `data/external/RWTH-2020-11307`
- Download frequency reponse of speaker at different angles (non-normalized directivity): Put `data/processed/RL906-spatial-FR.sofa` from https://nx29486.your-storageshare.de/s/grfHSXN8bJdQCp8 in `data/external`
- Optional but recommended: 
  - Install Jupyter Lab plugin for runtime: https://github.com/deshaw/jupyterlab-execute-time
  - Automatically use local .venv as kernel for Jupyter Lab: https://github.com/goerz/python-localvenv-kernel
- Run `rye sync` to set up your venv
- For webMUSHRA:
  - Download [webMUSHRA 1.4.3](https://github.com/audiolabs/webMUSHRA/releases/tag/v1.4.3) (release version with "dev") and unzip to `webMUSHRA-1.4.3` (if you want to, commit here)
  - Copy files from `webMUHRA-Customizations` to `webMUSHRA-1.4.3`, overwriting in case of conflicts


## Getting Started

- run `notebooks/12-create-free-field-equalized-hrtf.ipynb` to create normalized HRTFs
- The API flow with the main code parts is shown as a simple example in `notebooks/00-a-first-example.ipynb`
- The main processing steps for the thesis can be traced in the following files:
    - `23-e10-analysis.py` to run the SRIR analysis (approx. 30 mins)
    - `30-dropout-synthesis.py` for the reflection ablation study BRIR synthesis (approx. 6 hours)
    - `42-filtered-ds-synthesis.py` for the final BRIR synthesis comparing to baseline with only direct sound extrapolated (approx. 30 mins)  
    
    Carefully check these files before running them as they are long running. Use the provided variables to first only run one case, then extend to more, etc.


