# Chest-X-Ray
## A Tensorflow model to detect disease using chest X-rays    

**get_data.py** crawls data and json from [OpenI](https://openi.nlm.nih.gov/gridquery.php?q=pneumonias&it=x,xg&sub=x)   

**json_to_csv.py** create labels from labels.csv      

**resize_pic.py** resize picture to 224*224   

**shuffle_origin.py** reads data and labels from picture and csv(contains 6738 items), shuffles them and save as **train.npy** and **eval.npy**    

**read_array.py** contains functions reads data and labels from **train.npy** and **eval.npy**   

**nin.py** contains model and train routes    


Accuracy 60.55% Precision 67.02% Recall 73.58% F1score 70.03%  
