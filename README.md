# Bird Song Recognition

Proposed a classification model with ResNet as a backbone to classify bird calls into 100 categories and predict the bird species that appear in each test audio. 

Converted audio data into Mel Spectrograms before classification.

## Structure

```
-Bird Song Recognition
|---model			# Store models
|---data_csv.py		# Data preprocessing
|---dataset_load.py # For model training and testing
|---data_to_mel.py 	# Data preprocessing
|---handle_data.py	# Data preprocessing
|---main.py 		
|---mixup.py       	# Data augmentation
|---show.py
|---spectrum_pytroch.py		# Tool
|---t_optim.py		# Optimizer scheduled 
```





![!Aaron Swartz](https://github.com/supergirl-os/Bird-Song-Recognition/raw/main/img.png)

Figure 1 Visualizing the Mel feature map of the training set