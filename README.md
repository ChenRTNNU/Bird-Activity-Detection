# Data Augmentation for Bird Activity Detection

## Background

In this study, we aim to investigate various data augmentation methods to improve both in-domain and out-domain bird audio detection performance. Specifically, six data augmentation methods are used including (1) Time shift and Pitch shift, (2) Mixup, (3) SpecAugment, (4) Cut-instance Mixup, (5) Time-domain cross-condition data augmentation, (6) Constrained mini-batch based mixture masking. For the experiment, three datasets are used to design the experiments and verify our research conclusion. 

## Datasets

we use three public datasets for the experiment which are warblrb10k, freefield1010, and BirdVox-DCASE-20k. For all three datasets, each recording was captured under distinct environmental conditions, which makes it possible to investigate out-domain bird audio detection.

## Installation

### Requirements
- python 3.11
- pytorch 2.1.1
- librosa 0.10.1
- numpy 1.26.2
- scikit-learn 1.3.2
- tqdm 4.66.1
- pandas 2.1.3
- matplotlib 3.8.2

### Installation Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/ChenRTNNU/Bird-Activity-Detection.git
    ```

2. Navigate to the project directory: 
	```bash    
   cd yourProject   
	```

3. Install dependencies: 
    ```bash    
    pip install -r requirements.txt    
    ```
## Basic Usage

You can open the folder in the directory and open the .py file with the same name. Then change the following path in the code to your path. Last, you can run the file. For example, you can open the `baseline` folder and open `baseline.py`, then change the path. Last, you can run `baseline.py` to execute the baseline model. 

```python
csv_file_path1 = 'D:\\DeepLearning\\bird_activity_detection\\labels\\ff1010bird.csv'#change to your path
df1 = pd.read_csv(csv_file_path1)
csv_file_path2 = 'D:\\DeepLearning\\bird_activity_detection\\labels\\warblrb10k.csv'#change to your path
df2 = pd.read_csv(csv_file_path2)
#....
training_dataset_ff = AudioDataset(training_df1, data_folder1='D:\\DeepLearning\\bird_activity_detection\\audio\\ff1010bird')#change to your path
valid_dataset_ff = AudioDataset(valid_df1, data_folder1='D:\\DeepLearning\\bird_activity_detection\\audio\\ff1010bird')#change to your path
test_dataset_ff = AudioDataset(test_df1, data_folder1='D:\\DeepLearning\\bird_activity_detection\\audio\\ff1010bird')#change to your path
test_dataset_warb = AudioDataset(test_df2, data_folder1='D:\\DeepLearning\\bird_activity_detection\\audio\\warblrb10k')#change to your path
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
