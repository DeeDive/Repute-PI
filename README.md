# Repute-PI: An Unsupervised Framework for Online Reputation Monitor and Prediction with Review Information and Rating Dynamics

![Repute-PI Framework](images/reputepi.jpg)

## Run the model
To kick it off, change the phase to [1,2,3,4] in main.py, then

    python main.py

## Results

+ **Rating Module** - mitigate review-rating mismatches.
  + Review Embedding and Clustering. 

![t-SNE visualization of review embeddings](images/tsne.jpg)

+ + Effective Moving Rating with Review Correction

![Rating Correction](images/rating_correction.jpg)

+ **Prediction Module** - Reputation Prediction based on ARIMA

![Brand Reputation Prediction](images/prediction_example.jpg)

+ **Insight Module** - What customers care about most 

![Hair dryer example](images/hair_dryer_rating_5_title_wordcloud.png)
