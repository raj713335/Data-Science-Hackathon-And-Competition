 <p align="center">
    <img src="SRC/front.png", width="1400">
    <br>
    <sup><a href="https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-selling-price/problems/" target="_blank">Carnival Wars</a></sup>
</p>



<span class="light"></span>
<span class="dark"> Predict the price</span>
                        </div>
                        


<p>Halloween is a night of costumes, fun, and candy that takes place every year on October 31. On&nbsp;this day people dress up in various costumes that have a scary overtone and go&nbsp;trick-or-treating to gather candy.</p>

<p>This year, on Halloween,&nbsp;there is a&nbsp;carnival in your neighborhood. Besides the various games, there are also 50 stalls that are selling various products, which fall under&nbsp;various categories.</p>

<p>Your task is to predict the selling price of the products based on the provided features.&nbsp;</p>

<h2><strong>Data description</strong></h2>

<p>The data folder consists of the following three&nbsp;<strong>.csv</strong> files:</p>

<ul>
	<li><strong>train.csv</strong>:&nbsp;(<span class="mathjax-latex">\(6368 \times15\)</span>)</li>
	<li><strong>test.csv</strong>:&nbsp;(<span class="mathjax-latex">\(3430\times14\)</span>)</li>
	<li><strong>submission.csv</strong>: (<span class="mathjax-latex">\(5\times 2\)</span>)</li>
</ul>

<table border="1">
	<tbody>
		<tr>
			<td style="text-align:center"><strong>Column name&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;</strong></td>
			<td style="text-align:center"><strong>Description</strong></td>
		</tr>
		<tr>
			<td>
			<p>Product_id</p>
			</td>
			<td>
			<p>Unique ID of each product</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Stall_no</p>
			</td>
			<td>
			<p>Represents the number of stalls in the carnival (1-50)</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>instock_date</p>
			</td>
			<td>
			<p>Represents the date and time on&nbsp;which the product was bought</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Market_Category</p>
			</td>
			<td>
			<p>Represents the different market categories that the products belong to</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Customer_name</p>
			</td>
			<td>
			<p>Represents the names of the customers</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Loyalty_customer</p>
			</td>
			<td>
			<p>Represents if a customer is a loyal customer</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Product_Category</p>
			</td>
			<td>
			<p>Represents the 10 different product categories that the products belong to</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Grade</p>
			</td>
			<td>
			<p>Represents the quality of products</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Demand</p>
			</td>
			<td>
			<p>Represents the demand for the products being sold at&nbsp;the carnival</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Discount_avail</p>
			</td>
			<td>
			<p>Represents whether a product is being sold at a&nbsp;discount or not</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>charges_1</p>
			</td>
			<td>
			<p>Represents the&nbsp;types of charges applied on the products in the carnival</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>charges_2 (%)</p>
			</td>
			<td>
			<p>Represents the&nbsp;types of charges applied on the products in the carnival</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Minimum_price</p>
			</td>
			<td>
			<p>Represents the minimum price of a product</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Maximum_price</p>
			</td>
			<td>
			<p>Represents the maximum price of a product</p>
			</td>
		</tr>
		<tr>
			<td>
			<p>Selling_Price</p>
			</td>
			<td>
			<p>Represents the selling price of the product in the carnival</p>
			</td>
		</tr>
	</tbody>
</table>

<h2><strong>Evaluation metric</strong></h2>

<p><span class="mathjax-latex">\(score = {max(0,100 - RMSLE(actual\_values,predicted\_values))}\)</span></p>

<p><strong>Note</strong>: To avoid any discrepancies in the scoring, you must&nbsp;ensure that all the column values in the&nbsp;<strong>Product_id</strong>&nbsp;column in the file that you submit&nbsp;match the values in the&nbsp;<strong>test.csv</strong>&nbsp;provided.</p>

<p>&nbsp;</p>


# RESULT 86.86%

