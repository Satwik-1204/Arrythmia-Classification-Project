# ECG Arrhythmia Classification (CNN-LSTM)

A deep learning project implementing a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture in PyTorch to automatically detect and classify cardiac arrhythmias from Electrocardiogram (ECG) signals.

## Results
* **Overall Test Accuracy**: 95.06%
* **Ventricular Arrhythmia Accuracy**: 95.51% (Critical life-threatening class)
* **Macro-average AUC**: 0.9855

## Dataset
**MIT-BIH Arrhythmia Database**
* **Training Set**: 87,554 samples
* **Test Set**: 21,892 samples
* **Signal Length**: 187 time steps per heartbeat
* **Classes (5)**: Normal (83%), Supraventricular (3%), Ventricular (7%), Fusion (1%), Unclassifiable (6%).
* *Note: The severe 113:1 class imbalance was handled using dynamic class weighting in the PyTorch loss function.*

## Model Architecture
The hybrid model processes 1D medical time-series data by combining spatial feature extraction with sequential pattern recognition:

1. **CNN Block**: Three 1D Convolutional layers (scaling up to 128 filters) with MaxPooling, BatchNorm, and ReLU. This block reduces signal length while extracting local morphological features (e.g., specific PQRST wave anomalies).
2. **Bi-LSTM Block**: Two-layer Bidirectional LSTM (128 hidden units with 0.3 dropout). Captures long-term temporal dependencies and cardiac rhythm patterns by processing the sequence both forward and backward.
3. **Fully Connected Block**: Dense layers mapping the combined spatial-temporal features to the 5 output classes. Total parameters: ~500,000.

## Technologies Used
* **Deep Learning**: PyTorch (CUDA optimized)
* **Data Processing**: Pandas, NumPy, Scikit-learn
* **Visualization**: Matplotlib, Seaborn

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Satwik-1204/Arrythmia-Classification-Project.git
cd Arrhythmia_Classification_Project

# 2. Install core dependencies
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm

# 3. Run the notebook
jupyter notebook ECG_CNN_LSTM.ipynb
