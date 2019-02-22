# product-recommender
We use matrix completion to predict what products a customer is likely to have affinity for. Our data is an array of products and customers, with data on how many of each product the customers have purchased. We expect that some customers would have purchased more products, but were not aware of them. Matrix completion allows us to predict these affinities for use in targeted advertisement.

Matrix completion is a technique for filling in entries of an array which is only partially observed. When we see a customer has not bought a certain product, it could mean that they did not want to buy it, or that they were not aware of it. The optimal sales matrix, where all customers were aware of all products, will have more non-zero entries compared with our data matrix. Hence, we can treat our data matrix as a partially observed version of the optimal matrix, and use matrix completion to predict the optimal matrix. This will tell us which products each customer would be likely to purchase if they were aware of the products. 

The raw data is included in query_result.csv which we treat as a sparse array. It includes customer ID's, the number of each product they bought, as well as the total number of products they bought. We are looking for relationships between products which customers bought together. We don't want a few customers who bought many products to overwhelm all the other data, so we normalize the number of each product a customer purchased by their total.

From another point of view, we can think of this as a classification problem, where there are classes of customers who buy certain groups of products. We then want to classify each customer and recommend products from that group. This approach is only effective if there is a small number of classes, which is represented mathematically as a optimal sales matrix with low rank. For this reason, we round each type of product purchased to a multiple of 0.2 of that customer's total. This greatly reduces the rank of the data matrix and makes the classification of customers possible.

We first pre-process the data matrix to normalize the magnitude of entries, and reduce its rank.
The normed.csv is the normalized version of the query_result.csv, where also 0's are replaced by NaN's.
We will reconstruct entries which are NaN, but first create mask.csv which has entries of 1 in these places, and 0 where we have known data.

We use matrix completion algorithms from the fancyimpute package and dependencies. The solved.csv file contains the  reconstructed optimal sales matrix. Entries in solved.csv which were zeroes in the data matrix are products which we can recommend to individual customers with targeted advertisements.

Additional tests are included in the bootstrapping section of solve.py and in test.py.
