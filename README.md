### Zindi's South Africa "Spot the Crop XL Challenge"

Download the training, testing, and label data from [Radiant MLhub](https://mlhub.earth/10.34911/rdnt.j0co8q)

Run `python3 classify_crops.py -n 1000 -f .01 --path /path/to/datadirectory` 

to train a random forest classifier to recognize the 10 crop cover types, test 
the algorithm's accuracy, and print a confusion matrix. 

`-n 1000` randomly sample 1000 satellite images 

`-f .01` selects a fraction of 1% of the pixels from those 1000 satellite images
with which to train the algorithm

set `-svm True` to use a support vector machine instead of a random forest 
classifier

# crop types

        0: 'No Data'
        1: 'Lucerne/Medics'
        2: 'Planted pastures (perennial)'
        3: 'Fallow'
        4: 'Wine grapes
        5: 'Weeds'
        6: 'Small grain grazing'
        7: 'Wheat'
        8: 'Canola'
        9: 'Rooibos'
