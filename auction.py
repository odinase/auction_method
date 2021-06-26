import numpy as np


def unassigned_customers_exist(customers):
    return (customers < 0).any() # We use -1 for unassigned


def auction(A, eps=1e-3):
    m, n = A.shape
    unassigned_queue = np.arange(n)
    assigned_tracks = np.full(n, -1, dtype=int) # -1 indicates unassigned track
    prices = np.zeros(m, dtype=int)
    preffered_items = np.empty(n, dtype=int)

    while unassigned_queue.size > 0:
        t_star = int(unassigned_queue[0])
        unassigned_queue = unassigned_queue[1:] # Poor man's pop

        for k, rewards in zip(range(n), A.T):
            preffered_items[k] = int((rewards - prices).argmax())
        
        i_star = int(preffered_items[t_star])
        prev_owner, = np.where(assigned_tracks==i_star)
        assigned_tracks[t_star] = i_star
        if prev_owner.size > 0: # The item has a previous owner
            assert prev_owner.shape[0] == 1, f"multiple owners of same item, prev_owner = {prev_owner}"
            assigned_tracks[prev_owner] = -1
            unassigned_queue = np.append(unassigned_queue, prev_owner)

        values = np.delete(A.T[t_star] - prices, i_star)
        y = A[i_star, t_star] - values.max()
        prices[i_star] = prices[i_star] + y + eps

    return assigned_tracks


def calc_reward(problem_solution_pair):
    As, Ap = problem_solution_pair
    items = As
    customers = np.arange(As.shape[0])
    reward = Ap[items, customers].sum()

    return reward


def find_best_problem_solution_pair(problem_solution_set):
    best_reward = -np.inf
    best_pair = problem_solution_set[0]
    idx = 0
    for k, ps_pair in enumerate(problem_solution_set):
        curr_reward = calc_reward(ps_pair)
        if curr_reward > best_reward:
            best_pair = curr_reward
            idx = k

    return best_pair, idx

def murtys(A, N):
    m, n = A.shape
    As = auction(A)
    L = [(As, A)]
    R = []

    i = 0

    while i < N and len(L) > 0:
        M, k = find_best_problem_solution_pair(L)
        Ms, Mp = M
        R.append(Ms)
        L.pop(k)
        if len(R) == N:
            break

        for t in range(n):
            Qp = Mp.copy()
            # Qp[]




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
    reward = calc_reward((assignments, A))
    print(f"reward = {reward}")

    assignments = auction2(A, eps=0.01)
    reward = calc_reward2((assignments, A))
    print(f"reward = {reward}")

    for t, j in enumerate(assignments):
        print(f"a({t+1}) = {j+1}")

    