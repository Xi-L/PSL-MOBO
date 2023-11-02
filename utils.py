import torch


def rmse(target: torch.tensor, source: torch.tensor):
    """
    RMSE_x = \sqrt{\frac{\sum_{q=1}^{n_q}
    \lVert x_a^* - x_q \rVert^2}{n_q}}
    :param target: The target tensors
    :param source: The source tensors
    :return: scalar value to indicate how close the source related
    to the target tensors (the smaller, the better)
    """
    t_sample, t_dim = target.shape
    s_sample, s_dim = source.shape
    assert t_sample == s_sample and t_dim == s_dim

    ans = torch.sqrt(torch.sum(torch.pow(torch.norm(target - source, dim=1), 2))/t_sample)
    detail_ans = torch.pow(torch.norm(target - source, dim=1), 2)

    return ans.item(), detail_ans.detach()


def igd(front: torch.tensor, solution: torch.tensor):
    """
    IGD =\frac{1}{|Y^*|}\sum_{q=1}^{|Y^*|}
    \min\{\lVert y_q^*-y_1\rVert, ...,
    \lVert y_q^*-y_{n_q} \rVert\}
    :param front: The true pareto front or the ground truth x or y
    by the queried preference vector
    :param solution: The generated set of solutions to be measured
    by IGD metric.
    :return: scalar value to indicate how good the solution is
    (the smaller, the better)
    """
    front_s, front_dim = front.shape
    sol_s, sol_dim = solution.shape

    # the front and the solutions should be in same dimension
    assert front_dim == sol_dim

    # Compute vector-wise norm-2 distance
    tot_diff = front.unsqueeze(1) - solution.unsqueeze(0)
    tot_norm = torch.norm(tot_diff, dim=2)

    # tot_norm is in shape of [front_s, sol_s]
    tot_norm_min = torch.min(tot_norm, dim=1).values
    igd_ans = torch.sum(tot_norm_min)/front_s

    return igd_ans.item()
