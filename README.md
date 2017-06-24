# Chest-X-Ray
## A Tensorflow model to detect disease using chest X-rays  
**get_data.py** crawls data and json from [OpenI](https://openi.nlm.nih.gov/gridquery.php?q=pneumonias&it=x,xg&sub=x)  
**shuffle_origin.py** reads data and labels from picture and csv(contains 6738 items), shuffles them and save as **train.npy** and **eval.npy**  
**read_array.py** contains functions reads data and labels from **train.npy** and **eval.npy**  
**nin.py** contains model and train routes  
Testing Accuracy= 0.65361, Precision = 0.64672, Recall = 0.76303, f1_score = 0.70008
