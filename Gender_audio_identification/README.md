# Speaker gender identification project

This little project was made as a test work for one company.
The task was to train model to classify gender of the speaker.

All audio files are in .wav format with sample rate 8000 Hz, 16 bit.

## Getting Started

Main scripts are "workflow.py" and "gather_data.py".
Run workflow.py to see how the selected algorithm classifies speakers.

If you want to test different models or different parameters, then refer to "model_selection.py".
GridSearchCV is ready to go.

### Prerequisites

Code was written via Python 3.8.2 but should also work on any Python 3+ version.

All packages required are listed in requirements.txt, 
so use command "pip install -r requirements.txt" to install them.


## Authors

* **Alex Kuznetsov** - [Aleks-me](https://github.com/Aleks-me)
