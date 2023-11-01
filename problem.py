import torch
import numpy as np

DTLZ_pref_list = [62, 18, 12, 9, 9]


def generate_norm(source_tensor: torch.tensor):
    norm_tensor = torch.norm(source_tensor, dim=1).unsqueeze(1)
    target_tensor = source_tensor / norm_tensor
    return target_tensor


def ref_points(obj_num: int, granularity: int, if_norm: bool = True):
    """
    Return the reference tuple for decomposition-based methods
    like MOEA/D or NSGA-III
    :param obj_num: the objective size
    :param granularity: parameter H such that x_1 + x_2 +
    ... + x_obj_num = H
    :param if_norm: parameter boolean to indicate different modes
    :return: A torch.tensor in shape of [C(obj_num+H-1, H)]
    """

    # We solve this problem by DP-like algorithm
    dp_list = []
    for i in range(granularity + 1):
        dp_list.append(torch.tensor([i]).unsqueeze(0))

    for i in range(2, obj_num + 1):
        for j in range(granularity + 1):
            if j == 0:
                # prefix a zero simply
                dp_list[j] = torch.cat((torch.zeros_like(dp_list[j])[:, 0].unsqueeze(1), dp_list[j]), dim=1)
            else:
                # first prefix a zero simply
                dp_list[j] = torch.cat((torch.zeros_like(dp_list[j])[:, 0].unsqueeze(1), dp_list[j]), dim=1)
                # then plus one based on dp_list[j-1]
                dp_tmp = torch.zeros_like(dp_list[j-1])
                dp_tmp[:, 0] = 1
                dp_tmp = dp_tmp + dp_list[j-1]
                dp_list[j] = torch.cat((dp_list[j], dp_tmp), dim=0)

        # DEBUG:
        # print("shape {} in iteration {}.".format(dp_list[-1].shape, i))

    dp_list[-1] = dp_list[-1] / granularity

    if if_norm:
        return dp_list[-1]/torch.norm(dp_list[-1], dim=1).unsqueeze(1)
    else:
        return dp_list[-1]


def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'f1': F1,
        'f2': F2,
        'f3': F3,
        'f4': F4,
        'f5': F5,
        'f6': F6,
        'vlmop1': VLMOP1,
        'vlmop2': VLMOP2,
        'vlmop3': VLMOP3,
        'dtlz2': DTLZ2,
        're21': RE21,
        're23': RE23,
        're33': RE33,
        're36': RE36,
        're37': RE37,
        'mdtlz1_3_1': mDTLZ1(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz1_3_2': mDTLZ1(m=3, n=6, s=0.85, p=0.5, p_ind=1),
        'mdtlz1_3_3': mDTLZ1(m=3, n=6, s=0.70, p=0.5, p_ind=2),
        'mdtlz1_4_1': mDTLZ1(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz1_4_2': mDTLZ1(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'mdtlz1_4_3': mDTLZ1(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'mdtlz1_4_4': mDTLZ1(m=3, n=6, s=0.25, p=0.5, p_ind=3),
        'mdtlz2_3_1': mDTLZ2(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz2_3_2': mDTLZ2(m=3, n=6, s=0.85, p=0.5, p_ind=1),
        'mdtlz2_3_3': mDTLZ2(m=3, n=6, s=0.70, p=0.5, p_ind=2),
        'mdtlz2_4_1': mDTLZ2(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz2_4_2': mDTLZ2(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'mdtlz2_4_3': mDTLZ2(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'mdtlz2_4_4': mDTLZ2(m=3, n=6, s=0.25, p=0.5, p_ind=3),
        'mdtlz3_3_1': mDTLZ3(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz3_3_2': mDTLZ3(m=3, n=6, s=0.85, p=0.5, p_ind=1),
        'mdtlz3_3_3': mDTLZ3(m=3, n=6, s=0.70, p=0.5, p_ind=2),
        'mdtlz3_4_1': mDTLZ3(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz3_4_2': mDTLZ3(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'mdtlz3_4_3': mDTLZ3(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'mdtlz3_4_4': mDTLZ3(m=3, n=6, s=0.25, p=0.5, p_ind=3),

 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name]


class mDTLZ1:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "mDTLZ1"
        self.nadir_point = [5, 5, 5]
        if p_ind == 0:
            self.p_vec = None

        else:
            self.p_vec = torch.arange(1, self.k+1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)
        # in the biased searching space we need to guarantee
        # the biased transformation can obtain the uniform original counterpart
        x_I = torch.pow(x_I, 1.0/self.s)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        g_a, g_b, g_c = 1, 1, 2
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p_vec))

        g_result_inter = torch.sum(g_result_1 - g_result_2, dim=1).unsqueeze(1)
        g_result = g_a * (self.k + g_result_inter)
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s)
        sample_prod = torch.cumprod(x_new_I, dim=1)
        sample_minus = 1 - x_new_I
        sample_ones = torch.ones(sample_size, 1)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = 0.5 * torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None,
               if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            # print("The g result here is {} with X {}.".format(g_result, x_II))
            return (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\nParam p_ind is {}\n".format(self.m, self.n,
                                                                           self.s, self.p,
                                                                           self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = 0.5 * ref_vec
        return ref_vec, obj_vec


class mDTLZ2:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "mDTLZ2"
        self.nadir_point = [3.5, 3.5, 3.5]
        if p_ind == 0:
            self.p_vec = None
        else:
            self.p_vec = torch.arange(1, self.k+1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            # g_result_2 = torch.cos(20 * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec, 2)
            # g_result_2 = torch.cos(20 * torch.pi * (x_II - self.p_vec))
        g_result_inter = torch.sum(g_result_1, dim=1).unsqueeze(1)
        g_result = g_result_inter
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s) * torch.pi / 2
        sample_prod = torch.cumprod(torch.cos(x_new_I), dim=1)
        sample_minus = torch.sin(x_new_I)
        sample_ones = torch.ones(sample_size, 1)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None, if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            return (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\n".format(self.m, self.n,
                                                        self.s, self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = generate_norm(ref_vec)
        # obj vectors norm equals to 1
        return ref_vec, obj_vec


class mDTLZ3:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "mDTLZ3"
        self.nadir_point = [3.5, 3.5, 3.5]
        if p_ind == 0:
            self.p_vec = None
        else:
            self.p_vec = torch.arange(1, self.k+1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        g_a, g_b, g_c = 0.1, 0.1, 2
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p_vec))
        g_result_inter = torch.sum(g_result_1 - g_result_2, dim=1).unsqueeze(1)
        g_result = g_a * (self.k + g_result_inter)
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s) * torch.pi / 2
        sample_prod = torch.cumprod(torch.cos(x_new_I), dim=1)
        sample_minus = torch.sin(x_new_I)
        sample_ones = torch.ones(sample_size, 1)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None, if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, True returns Pareto front, False returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            return (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\n".format(self.m, self.n,
                                                        self.s, self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = generate_norm(ref_vec)
        # obj vectors norm equals to 1
        return ref_vec, obj_vec


class F1():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 =  0.0
        count1 = count2 =  0.0
            
        for i in range(2,n+1):
            yi    = x[:,i-1] - torch.pow(2 * x[:,0] - 1, 2)
            yi    = yi * yi
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
        

class F2():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 =  0.0
        count1 = count2 =  0.0
            
        for i in range(2,n+1):
            theta = 1.0 + 3.0*(i-2)/(n - 2)
            yi    = x[:,i-1] - torch.pow(x[:,0], 0.5*theta)
            yi    = yi * yi
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1 * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class F3():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0.0
        count1 = count2 = 0.0
        
        for i in range(2,n+1):
            xi = x[:,i-1]
            yi = xi - (torch.sin(4.0*np.pi* x[:,0]  + i*np.pi/n) + 1) / 2
            yi = yi * yi 
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0])) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class F4():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * torch.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8* x[:,0] * torch.cos(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class F5():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * torch.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:,0] * torch.cos((4.0*np.pi*x[:,0] + i*np.pi/n)/3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class F6():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
      
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - (0.3 * x[:,0] ** 2 * torch.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * torch.sin(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:,0] ** 2 * torch.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * torch.cos(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs



class VLMOP1():
    def __init__(self, n_dim = 1):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0]).float()
        self.ubound = torch.tensor([4.0]).float()
        self.nadir_point = [4, 4]
       
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 = torch.pow(x[:,0], 2)
        f2 = torch.pow(x[:,0] - 2, 2)
     
        objs = torch.stack([f1,f2]).T
        
        return objs
    

class VLMOP2():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).float()
        self.ubound = torch.tensor([2.0, 2.0,2.0, 2.0,2.0, 2.0]).float()
        self.nadir_point = [1, 1]
       
    def evaluate(self, x):
        
        n = x.shape[1]
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n))**2, axis = 1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n))**2, axis = 1))
     
        objs = torch.stack([f1,f2]).T
        
        return objs
    

class VLMOP3():
    def __init__(self, n_dim = 2):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([-3.0, -3.0]).float()
        self.ubound = torch.tensor([3.0, 3.0]).float()
        self.nadir_point = [10,60,1]
       
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        x1, x2 = x[:, 0], x[:, 1]
    
        f1 = 0.5 * (x1 ** 2 + x2 ** 2) + torch.sin(x1 ** 2 + x2 ** 2)
        f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8 + (x1 - x2 + 1) ** 2 / 27 + 15
        f3 = 1 / (x1 ** 2 + x2 ** 2 + 1) - 1.1 * torch.exp(-x1 ** 2 - x2 ** 2)
     
        objs = torch.stack([f1,f2,f3]).T
        
        return objs

class DTLZ2():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1]
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(2,n)]), axis = 0)
        g = sum1
        
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2)
        f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f3 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1,f2, f3]).T
        
        return objs
    
class RE21():
    def __init__(self, n_dim = 4):
        
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        
    def evaluate(self, x):
        
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 =  L * ((2 * x[:,0]) + np.sqrt(2.0) * x[:,1] + torch.sqrt(x[:,2]) + x[:,3])
        f2 =  ((F * L) / E) * ((2.0 / x[:,0]) + (2.0 * np.sqrt(2.0) / x[:,1]) - (2.0 * np.sqrt(2.0) / x[:,2]) + (2.0 /  x[:,3]))
        
        f1 = f1 
        f2 = f2 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class RE23():
    def __init__(self, n_dim = 4):
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 1, 10,10]).float()
        self.ubound = torch.tensor([100, 100, 200, 240]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = 0.0625 * torch.round(x[:,0])  
        x2 = 0.0625 * torch.round(x[:,1])  
        x3 = x[:,2]
        x4 = x[:,3]
        
        #First original objective function
        f1 = (0.6224 * x1 * x3* x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = f1.float()
        
        # Original constraint functions 	
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0/3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        
        
        g = torch.stack([g1,g2,g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
         
        f2 = torch.sum(g, axis = 0).to(torch.float64)
        
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class RE33():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        
        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
    
        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow((x2 * x2 - x1 * x1), 2)
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
                        
        g = torch.stack([g1,g2,g3,g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis = 0).float() 
        
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs
    
class RE36():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([12, 12, 12, 12]).float()
        self.ubound = torch.tensor([60, 60, 60, 60]).float()
        self.nadir_point = [5.931, 56.0, 0.355720675227]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]

        # First original objective function
        f1 = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = torch.stack([x1, x2, x3, x4])
        f2 = torch.max(l, dim = 0)[0]
        
        g1 = 0.5 - (f1 / 6.931)   
        
        g = torch.stack([g1])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)                
        f3 = g[0]
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs
    
class RE37():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        xAlpha = x[:,0]
        xHA = x[:,1]
        xOA = x[:,2]
        xOPTT = x[:,3]
 
        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)
 
         
        objs = torch.stack([f1,f2,f3]).T
        
        return objs