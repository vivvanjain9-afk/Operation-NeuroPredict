# Operation NeuroPredict

An AI-optimized computational model for predicting OPTN-mutation based ALS progression and personalized therapeutic optimization.

## Features

- Advanced ALS progression modeling
- OPTN gene variant analysis
- Biomarker-driven compound selection
- Personalized treatment protocol optimization
- Real-time disease trajectory visualization
- Cost-effectiveness analysis

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vivvanjain9-afk/Operation-NeuroPredict.git
cd Operation-NeuroPredict
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install streamlit numpy pandas plotly
```

## Running Locally

Start the Streamlit app:

```bash
streamlit run code.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Input Patient Data**: Adjust biomarkers using the sliders and dropdowns
   - Demographics (age, neurofilament levels, oxidative stress)
   - Functional assessment (FVC, ALSFRS-R, King's stage)
   - Progression rate and OPTN mutation type

2. **Generate Predictions**: Click "Generate Prediction" to see individual treatment effects

3. **Find Optimal Protocol**: Click "Find Optimal Protocol" to test all combinations and find the best treatment strategy

## Deployment

### Deploy on Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app" → Select this repository
5. Choose `code.py` as the main file
6. Click "Deploy"

Your app will be live at: `https://vivvanjain9-afk-operation-neuropredict-xxxxx.streamlit.app`

## Scientific Background

This model incorporates:
- OPTN mutation data from clinical literature (Maruyama et al. 2010, Feng et al. 2019)
- ALS progression statistics from PRO-ACT database
- Therapeutic efficacy data from published trials
- Real-world patient biomarker distributions

## License

Science Fair Project
