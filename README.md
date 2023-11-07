# parkinsons-fog-net
This project explores deep learning techniques to predict the occurrence of Freezing of Gait (FOG) events in individuals with Parkinson's Disease using a novel Multimodal Dataset of Freezing of Gait in Parkinson's Disease. The goal is to identify FOG events from time-series biometric data (EEG, EMG, ECG, skin conductance) by implementing and evaluating various deep learning models. The best-performing model, InceptionTime, achieved a test accuracy of 94%. The project is implemented utilizing TSAI, a state-of-the-art deep learning library dedicated to time-series problems.
[Link to the full paper](https://drive.google.com/file/d/115rvoGYdmdr8SWZzlCayyIkfFySbQ4lA/view?usp=sharing)

## Data
A public dataset regarding the detection of Freezing of Gait using physiological data was released in late 2022 [7]. This dataset contains readings from various sensors reading electroencephalogram (EEG), electromyogram (EMG), gait acceleration (ACC), and skin conductance (SC) from 12 patients with Parkinson’s Disease, as well as if Freezing of Gait is detected within a specific timeframe.

![](/results/FOGVis.png)

## Project Structure
The repository is structured as follows:
```
data/: Contains the raw datasets used for training and evaluation.
models/: Holds the trained models and their weights for future reference or additional training.
notebooks/: Jupyter notebooks detailing the experiments, model training, and evaluation process.
src/: Source code for data preprocessing, model implementation, and utility functions.
results/: Performance metrics, graphs, and confusion matrices from model evaluations.
requirements.txt: Python dependencies required to run the codebase.
```

## Data Preprocessing
Data preprocessing is a critical step in this project. The provided dataset in 2D array format (features x timesteps) was batched into sequences and transformed into a 3D array (samples x features x timesteps) to be compatible with TSAI. Each sequence is labeled based on the proportion of FOG labels within its timesteps, with a set threshold to determine the label for the sequence.

## Model Implementation
We explored several architectures:
* LSTM (Baseline): Long short-term memory
* GRU: Gated recurrent unit network
* InceptionTime (Best performing): Ensemble of CNNs to identify local and global shape patterns
* TCN: Temporal convolutional network
InceptionTime, a model designed specifically for time-series data, was chosen due to its excellent performance. We utilized TSAI's Learner class to facilitate model training and evaluation. The library's high-level abstractions simplified the experimentation with various architectures and hyperparameters.

## Results
The models' performances, ranked from best to worst, are as follows:
* InceptionTime: 94%
* LSTM: 91%
* TCN: 90%
* GRU: 88%
The InceptionTime model outperformed other architectures, providing a high accuracy rate and proving to be an effective tool for FOG prediction.

![](/results/InceptionTimeResults.png)

## To use the code in this repository:
* Clone the repo: `git clone https://github.com/username/project-name.git`
* Install dependencies: `pip install -r requirements.txt`
* Navigate to the notebooks/ directory and run the Jupyter notebooks to train models and visualize results.

## Acknowledgments
Multimodal Dataset of Freezing of Gait in Parkinson's Disease.
* Zhang W, Yang Z, Li H, Huang D, Wang L, Wei Y, Zhang L, Ma L, Feng H, Pan J, Guo Y, Chan P. Multimodal Data for the Detection of Freezing of Gait in Parkinson's Disease. Sci Data. 2022 Oct 7;9(1):606. doi: 10.1038/s41597-022-01713-8. PMID: 36207427; PMCID: PMC9546845.
* The TSAI team for their comprehensive time-series deep learning library.

