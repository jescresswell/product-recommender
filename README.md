# recommender
Matrix completion utilities and testing

Requires the fancyimpute package and dependencies

The query_result.csv is the data received from SQL query

The normed.csv is the pre-processed version of the query_result where 0's are replaced by nans

The mask.csv has enteries of 1 where we reconstructed the data, and 0 where we had known data

The solved.csv is the fully reconstructed matrix
