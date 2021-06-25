import numpy as np

def unassigned_customers_exist(customers):
    return (customers < 0).any() # We use -1 for unassigned


def auction(A, eps=1e-3):
    n = A.shape[1]
    m = A.shape[0]
    unassigned_queue = np.arange(n)
    assigned_items = np.full(m, -1, dtype=int) # -1 indicates unassigned item (measurements)
    prices = np.zeros(m, dtype=int)
    preffered_items = np.empty(n, dtype=int)

    while unassigned_queue.size > 0:
        t_star = int(unassigned_queue[0])
        unassigned_queue = unassigned_queue[1:] # Poor man's pop

        for k, rewards in zip(range(n), A.T):
            preffered_items[k] = int((rewards - prices).argmax())
        
        i_star = int(preffered_items[t_star])
        prev_owner, = np.where(assigned_items==t_star)
        assigned_items[i_star] = t_star
        if prev_owner.size > 0: # The item has a previous owner
            assert prev_owner.shape[0] == 1, f"multiple owners of same item, prev_owner = {prev_owner}"
            assigned_items[prev_owner] = -1
            unassigned_queue = np.append(unassigned_queue, prev_owner)

        values = np.delete(A.T[t_star] - prices, i_star)
        y = A[i_star, t_star] - values.max()
        prices[i_star] = prices[i_star] + y + eps

    return assigned_items       

def calc_reward(problem_solution_pair):
    As, Ap = problem_solution_pair
    items, = np.where(As != -1)
    customers = As[items]
    reward = Ap[items, customers].sum()

    return reward

def murtys(A, N):
    As = auction(A)
    L = [(As, A)]
    R = []

    i = 0

    while i < N and len(L) > 0:



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

    assignments = auction(A, eps=0.01)
    associated_measurements, = np.where(assignments != -1)
    associated_tracks = assignments[associated_measurements]
    reward = A[associated_measurements, associated_tracks].sum()
    print(f"reward = {reward}")

    for j, t in enumerate(assignments):
        print(f"a({t+1}) = {j+1}")

    