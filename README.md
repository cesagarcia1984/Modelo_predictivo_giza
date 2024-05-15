Gora ML Competition 

Gora (formerly RociFi) uses a credit risk metric which reflects a wallet's likelihood of defaulting on their debts by analyzing DeFi transaction history and behavior. The task of the competition is to build models which predict the likelihood and the ratio of individual liquidations, in comparison to the amount borrowed [total_liquidation_to_total_borrow].

Data Collection

Wallet’s transaction (borrow/lend/etc) history for the following lending protocols: 
Aave
Compound
Cream
RociFi
Venus
MakerDAO
GMX
Radiant

Transaction history on following chains: 
Ethereum (full transaction history)
Arbitrum (protocol specific history)
Fantom (protocol specific history)
Polygon (protocol specific history)
Optimism (protocol specific history)
BSC (protocol specific history)
Avalanche (protocol specific history)

Historical token prices for conversion to USDT on transaction time 

Lending protocols data can be fetched from their Subgraphs. Pricing data can be fetched from Coingecko or similar service.  



How to load the dataset:

Install giza-datasets (version 0.2.2)
In your python environment, instantiate a loader object and load the “gora-competition-training” dataset.
 

import certifi
import os
os.environ['SSL-CERT_FILE'] = certifi.where()
from giza_datasets import DatasetsLoader


loader = DatasetsLoader
df = loader.load("gora-competition-training")



List of features 

There are 5 groups of features: 
Borrow
Repay
Deposit
Redeem
Liquidation
Derived

Borrow and repay are self-explanatory, deposit means deposit of collateral, redeem is collateral withdrawal. Derived is a feature of a feature.

All amounts should be denominated in USD.

Input Features -  [These are the features that can be used as input for your prediction model] : 


total_borrow: total borrowed amount 
address: wallet address
count_borrow: borrow count 
avg_borrow_amount: average borrowed amount
std_borrow_amount: borrowed amount, standard deviation 
first_borrow_date: date of the first borrow transaction
token_borrow_model: borrowed token
borrow_amount_cv: borrowed amount, variation 
total_repay: total repaid amount
count_repay: repayment count
avg_repay_amount: average repaid amount 
std_repay_amount: repaid amount, standard deviation
repay_amount_cv: repaid amount, variation
total_deposit: total deposit amount 
count_deposit: deposit count 
avg_deposit_amount: average deposit amount
std_deposit_amount: deposit amount, standard deviation
deposit_amount_cv: deposit amount, variation 
total_redeem: total redeemed amount 
count_redeem: redeeming count 
avg_redeem_amount: average redeemed amount
std_redeem_amount: redeemed amount, standard deviation
redeem_amount_cv: redeemed amount, variation
days_since_first_borrow': days since first borrow transaction 
net_outstanding: total debt 
int_paid: same as total_repay
net_deposits: same as total_deposit
count_repays_to_count_borrows: count_repays / count_borrows 
avg_repay_to_avg_borrow: avg_repay / avg_borrow
net_outstanding_to_total_borrowed: net_outstanding / total_borrowed',
net_outstanding_to_total_repaid: net_outstanding / total_repaid',
count_redeems_to_count_deposits: count_redeems / count_deposits',
total_redeemed_to_total_deposits: total_redeemed / total_deposits',
avg_redeem_to_avg_deposit: avg_redeem / avg_deposit',
net_deposits_to_total_deposits: net_deposits / total_deposits',
net_deposits_to_total_redeemed: net_deposits / total_redeemed',
dex_total_sum_added : total liquidity added to dexes
dex_total_sum_removed : total liquidity removed from dexes
dex_total_sum_swapped :	total liquidity swapped in dexes
calc_start_time:
added_at :

  
Target Feature -  [This feature will be the prediction of the model] : 

Total_liquidation_to_total_borrow:  total_liquidation/ total_borrow

Supplementary Target Features-  [Additional features about liquidation, can be useful for analytics, DO NOT use these features for prediction] : 

total_liquidation: total liquidated amount
count_liquidation: liquidations count
avg_liquidation_amount: average liquidated amount
std_liquidation_amount': liquidated amount, standard deviation
liquidation_amount_cv: liquidated amount, variation 
avg_liquidation_to_avg_borrow: avg_liquidation / avg_borrow'
liquidated: boolean, did the wallet ever got liquidated



Project Submission

To submit your project, please create a github repo with the following: 

1- Jupyter Notebook (or a similar python environment), where you develop the model.

We will score Notebooks with relevant data exploration steps such as:  Evaluation Matrix, Visualizations or similar Data Analysis processes more. Additionally, please make sure to write documentation for all the steps being taken.


2- An application that takes the same input format as the training set and gives as output the prediction. This is very important, as we will test your models with our test sets using this application. If your application does not work, we cannot score your model. If your model includes some preprocessing, make sure your application includes the same preprocessing steps. Your models will be tested by  RMSE score. [We have decided to allow people to access the unlabeled version of the test set, please make the predictions on the test set in your notebook and save the results as a .csv or .npy file]

3- A readme of the project, explaining the relevant findings and decisions for your model.

