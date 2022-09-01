import pickle
import operator
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

IT_NUM = 100




def normalizeDataset(X):
    '''
    Normalize each row of the dataset X

    :param X: dataset
    :return: dataset normalized by row
    '''
    mods = np.linalg.norm(X, axis=1)
    return X / mods[:, np.newaxis]


def populate1():
    cnt = 0
    items = dict()
    customer = dict()
    fp = open('transactions.csv')
    for line in fp:
        cnt += 1
        if cnt == 1:
            continue
        cust = line.split(',')[0]
        it = line.split(',')[3]
        if it not in items:
            items[it] = set(cust)
        else:
            items[it].add(cust)
    for key, val in items.items():
        items[key] = len(val)
    sorted_items = sorted(items.items(), key=operator.itemgetter(1))
    freq_items = [tup[0] for tup in sorted_items[-IT_NUM:]]
    cnt = 0
    freq_items_dict = dict()
    for it in freq_items:
        freq_items_dict[it] = cnt
        cnt += 1
    print(freq_items, len(freq_items))

    cnt = 0
    fp = open('transactions.csv')
    for line in fp:
        cnt += 1
        if cnt == 1:
            continue
        cust = line.split(',')[0]
        it = line.split(',')[3]
        if it in freq_items:
            if cust not in customer:
                customer[cust] = [0] * IT_NUM
            customer[cust][freq_items_dict[it]] = 1

    print(len(customer))
    pickle.dump([customer, freq_items_dict], open('transactions_dump.p', 'wb'))


def populate():
    '''
    Generate *transactions_dump.p* from *transactions.csv*.
    Only the first IT_NUM items are used. All other items ar totally ignored.
    Only the first 250k customers are considered and only included in the final dataset if the purchased one of the first IT_NUM items.

    The data is saved as a pickeled array of two dics.
        The first dict maps the customers to an array of integers. The integer of a index is one if the item "of this index" was bought by the customer.
        The second dict maps an item to "its" assigned index

    :return: None
    '''
    print('-' * 10, 'populate', '-' * 10)
    fp = open('transactions.csv')
    cnt, cust_cnt, it_cnt = 0, 0, 0
    # Dict mapping item to index of item in customer-values
    items = dict()
    # Dict mapping customer to array representing bought items by index
    customer = dict()
    last_cust = ''
    for line in fp:
        cnt += 1
        # Jump over first line (because headers)
        if cnt == 1:
            continue
        cust = line.split(',')[0]
        it = line.split(',')[3]
        # Adds item if the maximum number of used items (IT_NUM) has not been reached
        if it not in items and it_cnt < IT_NUM:
            items[it] = it_cnt
            it_cnt += 1
        # Add a new customer if necessary
        if cust not in customer:
            customer[cust] = [0] * IT_NUM
            cust_cnt += 1
            last_cust = cust
            # Print status-update every 10k lines
            if cust_cnt % 1000 == 0:
                print(f'Customer: {cust_cnt}, Line: {cnt}, Items: {it_cnt}')
        # Stop after 250k customers
        if cust_cnt > 250000:
            break
        # Register item wih customer if item is one of the first hundred items
        if it in items:
            customer[cust][items[it]] = 1
    # Loop breaks when last customer is first added => Last customer only has one item and should be deleted
    del customer[last_cust]
    print(f'Finished collecting the data of {len(customer)} customers on {len(items)} items.')

    # Remove all customers without purchase (of an item within des first IT_NUM items)
    print('Remove Customers without recorded purchases...')
    no_purchase = []
    for key, val in customer.items():
        if 1 not in val:
            no_purchase.append(key)
    for cus in no_purchase:
        del customer[cus]
    print(f'Saving data on {len(customer)} remaining customers to "transactions_dump.p".')
    print(len(customer), len(items))
    # Save data
    pickle.dump([customer, items], open('transactions_dump.p', 'wb'))


def make_dataset():
    '''
    Clustering the customers in 100 clusters and using the cluster a customer is assigned to as label for later training.
    Items-data is disregarded
    Clustering algorithm used is KMeans by sklaearn

    Features are saved as a pickeled np.array in *purchase_100_features.p*
    Labels are saved as pickeled np.array in *purchase_100_labels.p*

    :return: None
    '''
    print('-' * 10, 'Converting and labeling data', '-' * 10)
    dataset = []
    customer, items = pickle.load(open('transactions_dump.p', 'rb'))
    for key, val in customer.items():
        dataset.append(val)
    dataset = np.array(dataset)
    dataset = normalizeDataset(dataset)
    print(dataset.shape)
    print('Save data as np.array...')
    pickle.dump(dataset, open('purchase_100_features.p', 'wb'))
    print('Clustering data...')
    X = KMeans(n_clusters=100, random_state=0).fit(dataset)
    print(f'Saving labels.... ({len(np.unique(X.labels_))} unique labels)')
    pickle.dump(X.labels_, open('purchase_100_labels.p', 'wb'))


def main():
    # Note: transactions.csv file can be downloaded from https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
    # populate1() # 100 'most' frequent items
    populate()  # first 100 frequent items -- generates the data set used in the experiments
    make_dataset()


if __name__ == '__main__':
    main()
