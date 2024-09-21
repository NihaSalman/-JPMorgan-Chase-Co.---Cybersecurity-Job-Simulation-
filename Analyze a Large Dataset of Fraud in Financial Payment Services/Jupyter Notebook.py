import pandas as pd
import matplotlib.pyplot as plt
import random

def exercise_0(file):
    #Reading the dataset (`transactions.csv`) as a Pandas dataframe.
    df = pd.read_csv(file)
    print(df.head())
    return df

def exercise_1(df):
    #Returning the column names as a list from the dataframe.
    column_names = [col for col in df.columns]
    return column_names

def exercise_2(df, k):
    #Returning the first k rows from the dataframe.
    return df.head(k)

def exercise_3(df, k):
    #Returning a random sample of k rows from the dataframe.
    return df.sample(k)

def exercise_4(df):
    #Obtaining unique values from the 'type' column
    unique_types = df['type'].unique()
    #Converting the unique values to a list using list comprehension
    unique_transactions = [item for item in unique_types]
    return unique_transactions

def exercise_5(df):
    #Counting the frequencies of each transaction destination
    frequency_counts = df['nameDest'].value_counts()
    #Getting the top 10 transaction destinations with their frequencies
    top_10_destinations = frequency_counts.head(10)
    return top_10_destinations

def exercise_6(df):
    #Filtering rows where fraud was detected
    fraud_detected = df[df['isFraud'] == 1]
    return fraud_detected

def exercise_7(df):
    #Grouping by the 'nameOrig' column and counting the number of unique 'nameDest' (destinations) per source
    result = df.groupby('nameOrig')['nameDest'].nunique()
    #Sort the result in descending order
    result = result.sort_values(ascending=False)
    return result.reset_index()

# Test exercises here
#Call exercise_0 
print("Exercise 0")
df = exercise_0('transactions.csv')
print("\n")

#Call exercise_1 
print("Exercise 1")
column_names = exercise_1(df)
print("Column names list:\n", column_names)
print("\n")

#Call exercise_2
print("Exercise 2")
k=11
first_k_rows = exercise_2(df, k)
print(first_k_rows)
print("\n")

#Call exercise_3
print("Exercise 3")
k = random.randint(1, len(df)) 
first_k_rows = exercise_3(df, k)
print(first_k_rows)
print("\n")

#Call exercise_4
print("Exercise 4")
unique_types = exercise_4(df)
print("Unique transaction:", unique_types)
print("\n")

#Call exercise_5
print("Exercise 5")
top_10_destinations = exercise_5(df)
print("Top 10 transaction destinations with frequencies:")
print("\n")
print(top_10_destinations)
print("\n")

#Call exercise_6
print("Exercise 6")
fraud_rows = exercise_6(df)
print("Rows with fraud detected:")
print("\n")
print(fraud_rows)
print("\n")

#Call exercise_7
print("Exercise 7")
distinct_destinations = exercise_7(df)
print("Number of distinct destinations each source has interacted with:")
print("\n")
print(distinct_destinations)
print("\n")



def visual_1(df):
    #note: I had to change bits of the code given in the notebook to better plot the graph
    def transaction_counts(df):
        #Counting the occurrences of each transaction type
        return df['type'].value_counts()

    def transaction_counts_split_by_fraud(df):
        #Useing pivot_table to count occurrences of each transaction type split by fraud
        return df.pivot_table(index='type', columns='isFraud', aggfunc='size', fill_value=0)

    fig, axs = plt.subplots(2, figsize=(10, 8)) 

    #Getting the transaction types in the order of the first chart
    transaction_order = transaction_counts(df).index

    #Transaction types bar chart
    transaction_counts(df).loc[transaction_order].plot(ax=axs[0], kind='bar')
    axs[0].set_title('Transaction Types Distribution', fontsize=14)
    axs[0].set_xlabel('Transaction Type', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)
    
    #Transaction types split by fraud bar chart using the same order of types as the first for better readability 
    transaction_counts_split_by_fraud(df).loc[transaction_order].plot(ax=axs[1], kind='bar')
    axs[1].set_title('Transaction Types Split by Fraud', fontsize=14)
    axs[1].set_xlabel('Transaction Type', fontsize=12)
    axs[1].set_ylabel('Frequency', fontsize=12)
    axs[1].set_yscale('log')
    
    fig.suptitle('Transaction Types Analysis', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for ax in axs:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)

    return '''Visual_1(df):
The charts show the distribution of transaction types and how they are split by fraud. 
The first chart shows a general view of the transaction types, 
while the second chart highlights the relationship between transaction types and fraud detection. 
In this case, fraud is detected primarily in the transaction types "CASH_OUT" and "Transfer".'''

visual_1(df)



def visual_2(df):
    def query(df):
        # Filtering for Cash Out transactions
        cash_out_df = df[df['type'] == 'CASH_OUT'].copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Calculating balance delta for origin and destination accounts
        cash_out_df.loc[:, 'orig_delta'] = cash_out_df['newbalanceOrig'] - cash_out_df['oldbalanceOrg']
        cash_out_df.loc[:, 'dest_delta'] = cash_out_df['newbalanceDest'] - cash_out_df['oldbalanceDest']
        
        return cash_out_df[['orig_delta', 'dest_delta']]
    
    # Generating scatter plot
    plot = query(df).plot.scatter(x='orig_delta', y='dest_delta')
    plot.set_title('Origin vs. Destination Account Balance Delta for Cash Out Transactions')
    plot.set_xlabel('Origin Account Balance Delta')
    plot.set_ylabel('Destination Account Balance Delta')
    
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)

    # Added gridlines for easier visualization
    plot.grid(True)

    return ('''Visual_2(df):
The scatter plot displays the relationship between the balance delta of the origin and destination accounts
for Cash Out transactions. Each point represents a transaction, with its position indicating the change
in balance for both accounts. This visualization helps in understanding how changes in the origin account balance
relate to changes in the destination account balance during Cash Out operations.''')

visual_2(df)



def exercise_custom(df):
    #Grouping by the fraud and flagged fraud columns
    flagged_fraud_counts = df.groupby(['isFraud', 'isFlaggedFraud']).size().reset_index(name='Count')
    return flagged_fraud_counts

def visual_custom(df):
    #Getting the counts of flagged and actual fraud from the function exercise_custom(df)
    flagged_fraud_counts = exercise_custom(df)
    #Creating a pivot table 
    pivot_table = flagged_fraud_counts.pivot(index='isFraud', columns='isFlaggedFraud', values='Count').fillna(0)

    #Plotting the results
    fig, ax = plt.subplots(figsize=(8, 6))
    
    #Seting logarithmic scale for better visibility of small values as the number of actual fraud is very small
    ax.set_yscale('log')

    #Plotting the data with different colors for non-flagged (red) and flagged (blue) fraud cases
    pivot_table.plot(kind='bar', stacked=True, ax=ax, color=['red', 'blue'])

    ax.set_title('Comparison of Fraud Flagging Effectiveness', fontsize=14)
    ax.set_xlabel('Actual Fraud (0: No, 1: Yes)', fontsize=12)
    ax.set_ylabel('Number of Transactions (Log Scale)', fontsize=12)

    #A custom legend with specific labels for colors
    ax.legend(['Not Flagged (Red)', 'Flagged Fraud (Blue)'], title='Flagged Fraud', fontsize=10)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    return '''visual_custom(df):
The chart compares the number of transactions that were flagged as fraudulent versus those that were actually fraudulent. 
The purpose of this chart is to compare the transactions with fraudulent transactions (isFraud) to see if the flagging 
mechanism is effective or if there are discrepancies. In this case out of 200000 data only 147 were actual fraud and out of
those 147 fraudulent data none were flagged as fraudulent. '''

visual_custom(df)


#Extra information for whoever reads my code
print('''For the custom exercise, I initially intended to create a bar chart to show the origin and destination accounts with the 
highest number of fraud incidents. However, most accounts had only committed fraud once, which rendered this chart less meaningful. 
In fraud detection, it's common for perpetrators to use multiple accounts to spread their fraudulent activities, 
making detection more challenging. Therefore, I decided to shift the focus of the chart to evaluate the effectiveness of the isFraud mechanism.''')