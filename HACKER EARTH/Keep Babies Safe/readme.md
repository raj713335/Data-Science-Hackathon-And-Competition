 <p align="center">
    <img src="SRC/front.png">
    <br>
    <sup><a href="https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-baby-safety/machine-learning/identify-baby-product-and-brand-7-6386aaf6/" target="_blank">Keep Babies Safe</a></sup>
</p>

 <hr class="hr prob-hr"/>
                        <div class="problem-title large-20 darker less-margin-2-bottom content">Keep babies safe</div>
                        <div class="content prob-title-content">
                            <div class="float-left light small">Max. score: 100</div>
                            
                           
<h1>Problem statement</h1>

<p>A renowned hypermarket, that has multiple stores, is all set to grow its business. As a part of its expansion, the owner plans to introduce an exclusive section for newborns and toddlers. To help him expand its business, teams across all stores have collated numerous images that consist of products and goods purchased for children.&nbsp;</p>

<p>You are a Machine Learning expert. Your task is to:</p>

<ul>
	<li>
	<p>Identify the product type [toys,consumer_products]&nbsp;</p>
	</li>
	<li>
	<p>Extract and tag brand names of these products from each image. In case if no brand names are mentioned, tag it as ‘Unnamed’.&nbsp;</p>
	</li>
</ul>

<p>The extracted brand names must be represented in the&nbsp;format provided in sample submission.</p>

<h1>Data</h1>

<p>The dataset consists of one folder and .csv file named images and test.csv respectively.</p>

<p>There are 1131 images in the images folder.</p>

<p>The test.csv file contains the name of the images for testing.</p>

<p><strong>Note</strong>: Only test data is provided.</p>

<table border="1" style="width:372px">
	<caption>Data Description</caption>
	<tbody>
		<tr>
			<td style="width:116px">
			<p><strong>Columns</strong></p>
			</td>
			<td style="width:242px">
			<p><strong>Description</strong></p>
			</td>
		</tr>
		<tr>
			<td style="width:116px">Image</td>
			<td style="width:242px">Name of image</td>
		</tr>
		<tr>
			<td style="width:116px">Class_of_image</td>
			<td style="width:242px">Class of image (target column) [toys,consumer_products]&nbsp;</td>
		</tr>
		<tr>
			<td style="width:116px">Brand_name</td>
			<td style="width:242px">Name of the brand (target column)</td>
		</tr>
	</tbody>
</table>

<h1>Submission format</h1>

<p>You are required to&nbsp;write your predictions in&nbsp;a .<strong>csv</strong> file and upload it by clicking on the&nbsp;<strong>Upload file</strong>.</p>

<table border="1" style="width:346px">
	<caption>sample_submission.csv</caption>
	<tbody>
		<tr>
			<td style="width:44px"><strong>Image</strong></td>
			<td style="width:143px"><strong>Class_of_image</strong></td>
			<td style="width:139px"><strong>Brand_name</strong></td>
		</tr>
		<tr>
			<td style="width:44px">1.jpg</td>
			<td style="width:143px">toys</td>
			<td style="width:139px">Unnamed</td>
		</tr>
		<tr>
			<td style="width:44px">2.jpg</td>
			<td style="width:143px">toys</td>
			<td style="width:139px">Unnamed</td>
		</tr>
		<tr>
			<td style="width:44px">3.jpg</td>
			<td style="width:143px">consumer_products</td>
			<td style="width:139px">Happy Cappy/Mustela</td>
		</tr>
		<tr>
			<td style="width:44px">4.jpg</td>
			<td style="width:143px">consumer_products</td>
			<td style="width:139px">Johnsons</td>
		</tr>
		<tr>
			<td style="width:44px">5.jpg</td>
			<td style="width:143px">toys</td>
			<td style="width:139px">Hot Wheels</td>
		</tr>
	</tbody>
</table>

<h1>Evaluation Criteria</h1>

<p><span class="mathjax-latex">\(s1 = f1\_score(actual['Class\_of\_image'],predicted['Class\_of\_image'],average = 'weighted')\\ s2 = len(actual['Brand\_name'].intersection(predicted['Brand\_name']))/len(actual['Brand\_name'].union(predicted['Brand\_name']))\\ score = 0.7*s1 + 0.3*s2\)</span></p>
                            
