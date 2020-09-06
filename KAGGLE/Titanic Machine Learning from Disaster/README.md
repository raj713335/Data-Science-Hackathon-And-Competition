<p align="center">
    <img src="front.png", width="1000">
    <br>
    <sup><a href="https://www.kaggle.com/c/titanic/overview" target="_blank">Janatahack-Healthcare-Analytics-II</a></sup>
</p>


# THE CHALLENGE


The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

Who should participate?
This hackathon is open for all data enthusiasts: Statisticians, Data scientists, Analysts, and Students






# What Data Will I Use in This Competition?


In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled `train.csv` and the other is titled `test.csv`.

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

The `test.csv` dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

Check out the “Data” tab to explore the datasets even further. Once you feel you’ve created a competitive model, submit it to Kaggle to see where your model stands on our leaderboard against other Kagglers.
















# Submission File Format:

You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:
<ul>
<li><b>PassengerId </b>(sorted in any order)</li>
<li><b>Survived </b>(contains your binary predictions: 1 for survived, 0 for deceased)</li>
</ul>


### Evaluation Metric
The evaluation metric for this hackathon is 100*Accuracy Score.



### Public and Private split
The public leaderboard is based on 40% of test data, while final rank would be decided on remaining 60% of test data (which is private leaderboard)

 


  
  
# HACKATHON URL

 https://www.kaggle.com/c/titanic/overview

