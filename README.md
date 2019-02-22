# product-recommender
We use matrix completion to predict what products a customer is likely to have affinity for. Our data is an array of products and customers, with data on how many of each product the customers have purchased. We expect that some customers would have purchased more products, but were not aware of them. Matrix completion allows us to predict these affinities for use in targeted advertisement.

Matrix completion is a technique for filling in entries of an array which is only partially observed. When we see a customer has not bought a certain product, it could mean that they did not want to buy it, or that they were not aware of it. The optimal sales matrix, where all customers were aware of all products, will have more non-zero entries compared with our data matrix. Hence, we can treat our data matrix as a partially observed version of the optimal matrix, and use matrix completion to predict the optimal matrix. This will tell us which products each customer would be likely to purchase if they were aware of the products. 

The raw data is included in query_result.csv.


Requires the fancyimpute package and dependencies

The normed.csv is the pre-processed version of the query_result where 0's are replaced by nans

The mask.csv has enteries of 1 where we reconstructed the data, and 0 where we had known data

The solved.csv is the fully reconstructed matrix
