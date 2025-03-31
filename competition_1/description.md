[Kobe Bryant Shot Selection](https://www.kaggle.com/competitions/kobe-bryant-shot-selection)    
## Description
Kobe Bryant marked his retirement from the NBA by scoring 60 points in his final game as a Los Angeles Laker on Wednesday, April 12, 2016. Drafted into the NBA at the age of 17, Kobe earned the sport’s highest accolades throughout his long career.

Using 20 years of data on Kobe's swishes and misses, can you predict which shots will find the bottom of the net? This competition is well suited for practicing classification basics, feature engineering, and time series analysis. Practice got Kobe an eight-figure contract and 5 championship rings. What will it get you?

## Acknowledgements
Kaggle is hosting this competition for the data science community to use for fun and education. For more data on Kobe and other NBA greats, visit stats.nba.com.

## Evaluation
Submissions are evaluated on the `log loss`.

## Submission File
For each missing shot_made_flag in the data set, you should predict a probability that Kobe made the field goal. The file should have a header and the following format:

```angular2svg
shot_id,shot_made_flag
1,0.5
8,0.5
17,0.5
etc.
```
