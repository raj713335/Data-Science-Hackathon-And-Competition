# FRIENDSHIP GOALS 

To celebrate this precious relationship of friendship on the upcoming International Friendship Day, we bring a deep learning challenge to you: #FriendshipGoals. The Anthropology department of an Ivy League school is planning to study the impact of friendship at different life stages. You have been hired as a deep learning specialist to assist their team in this project.

Your task is to build a deep learning model that analyzes an image of a gathering among friends, detects the age group of the gathering, and classifies them into groups of toddlers, teenagers, or adults.



<p>Your task is to build a deep learning model that analyzes an image of a gathering among friends, detects the age group of the gathering, and classifies them into groups of&nbsp;<strong><em>toddlers</em></strong>,&nbsp;<strong><em>teenagers</em></strong>, or&nbsp;<strong><em>adults</em></strong>.</p>

<h2>Data</h2>

<p><strong>Data description</strong></p>

<p>You must use an external dataset to train your model. The attached dataset link contains the sample data of each category [Toddler&nbsp;| Teenagers | Adults] and test data.</p>

<p><strong>Data files</strong></p>

<table border="1">
	<tbody>
		<tr>
			<td style="text-align:center"><strong>File name</strong></td>
			<td style="text-align:center"><strong>Description</strong></td>
		</tr>
		<tr>
			<td>Test.zip</td>
			<td>Contains image files to be classified</td>
		</tr>
		<tr>
			<td>Sample.zip</td>
			<td>Contains sample image files belonging to each category</td>
		</tr>
		<tr>
			<td>Test.csv</td>
			<td>Predictions file containing indices of test data and a blank target column</td>
		</tr>
		<tr>
			<td>sample_submission.csv</td>
			<td>Submission format that must be followed for uploading predictions</td>
		</tr>
	</tbody>
</table>

<p><strong>Data description</strong></p>

<table border="1">
	<tbody>
		<tr>
			<td style="text-align:center"><strong>Column name</strong></td>
			<td style="text-align:center"><strong>Description</strong></td>
		</tr>
		<tr>
			<td>Filename</td>
			<td>File name of test data image</td>
		</tr>
		<tr>
			<td>Category</td>
			<td>Target column [values: 'Toddler'<strong>/</strong>'Teenagers'<strong>/</strong>'Adults']</td>
		</tr>
	</tbody>
</table>

<h2>Submission format</h2>

<p>Please refer to&nbsp;<a href="https://s3-ap-southeast-1.amazonaws.com/he-public-data/Sample%20Submission15b2050.csv" target="_blank">sample_submission.csv</a>&nbsp;for more details</p>

<h2>Evaluation criteria</h2>

<p><span class="mathjax-latex">\(score=100∗recall\_score(actual\_values,predicted\_values)\)</span></p>

<p>&nbsp;</p></div>





# The eight categories of Indian classical dance are as follows:

`Manipuri`

`Bharatanatyam`

`Odissi`

`Kathakali`

`Kathak`

`Sattriya`

`Kuchipudi`

`Mohiniyattam`

# Data description

This data set consists of the following two columns:

Column Name	Description

Image	Name of Image

target	Category of Image `['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam']`

The data folder consists of two folders and two .csv files. The details are as follows:

train: Contains 364 images for 8 classes `['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam']`

test: Contains 156 images

train.csv: 364 x 2

test.csv: 156 x 1

# Submission format

You are required to write your predictions in a .csv file and upload it by clicking the Upload File button.

# Sample submission

Image,target

96.jpg,manipuri

163.jpg,bharatanatyam

450.jpg,odissi

219.jpg,kathakali

455.jpg,odissi

46.jpg,kathak

# Evaluation metric
\(score = {100* f1\_score(actual\_values,predicted\_values,average = 'weighted')}\)

# Note: To avoid any discrepancies in the scoring, ensure all the index column (Image) values in the submitted file match the values in the provided test.csv file.

Time Limit:	5.0 sec(s) for each input file.

Memory Limit:	256 MB

Source Limit:	1024 KB


# RESULT

<p align="center">
    <img src="back.png", width="1000">
    <br>
    <sup><a href="https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/problems/">Identify The Dance Form</a></sup>
</p>

247 out of 5846 TOP 4%



