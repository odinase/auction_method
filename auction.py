import numpy as np

def first_unassigned_customer(customers):
    return np.where(np.isnan(customers))[0][0]

def unassigned_customers_exist(customers):
    return np.isnan(customers).any()


def auction(A, eps=1e-3):
    n = A.shape[1]
    m = A.shape[0] - n
    customers = np.full(n, np.nan)
    prices = np.zeros(m)
    preffered_items = np.empty(n)

    while unassigned_customers_exist(customers):
        t_star = first_unassigned_customer(customers)

        for k, rewards in zip(range(n), A.T):
            preffered_items[k] = (rewards[:m] - prices).argmax()
        
        i_star = preffered_items[t_star]
        customers[t_star] = i_star
        



if __name__ == "__main__":
    A = np.array([
        [  -5.69,    5.37, -np.inf],
        [-np.inf,    -3.8,    6.58],
        [   4.78, -np.inf, -np.inf],
        [-np.inf,    5.36, -np.inf],
        [  -0.46, -np.inf, -np.inf],
        [-np.inf,   -0.52, -np.inf],
        [-np.inf, -np.inf,   -0.60]
    ])

