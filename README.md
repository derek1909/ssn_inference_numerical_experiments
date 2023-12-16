Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference
-------------------------------------------------------------------------------------------------

The following repository contains the code and parameters corresponding to the numerical experiments presented in:

Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2019). Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference. bioRxiv, 696088.

The repository consists of three parts:

- `GSM`) The code related to the generative model (the GSM), required to generate the stimuli and its corresponding target moments.
- `SSN`) The code related to the numerical experiments carried out with the already optimized network
- `GP`) The code related to the numerical experiments carried out with Gaussian processes.

The network parameters can be found at `SSN\parameter_files` and in the python file `SSN\parameters.py`

The optimizer can be found in a separate repository. See the fore-mentioned paper for a link to its current location.

# Terms of use

This repository is provided for reproducibility purposes only. Please contact the authors if you wish to use any parts of this code for other purposes.

# Requirements

`python`
`numpy`
`scipy`
`tensorflow 1.0` (for some scripts)
