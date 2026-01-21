def generate_archimedean_data(num_samples, alpha, theta):
    """
    Generates synthetic data with a non-Gaussian (Archimedean - Clayton) dependency.
    This function implements the Clayton copula sampling algorithm directly using PyTorch.

    Args:
        num_samples (int): The number of data points to generate.
        alpha (float): The true causal effect (ACE) to embed in the data.
        theta (float): The dependency parameter for the Clayton copula. theta > 0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated T_data and Y_data tensors.
    """
    # 1) Generate two independent uniform random variables, u1 and u2, on [0, 1].
    # These will serve as the independent sources of randomness for our system.
    u1 = torch.rand(num_samples, 1)
    u2 = torch.rand(num_samples, 1)

    # 2) Create a new uniform variable 'u2_given_u1' that is dependent on 'u1'
    #    via the Clayton copula's conditional sampling formula:
    #    C(u₂|u₁) = [u₁^(-θ) × (u₂^(-θ/(1+θ)) - 1) + 1]^(-1/θ)
    # This step transforms the independent u2 into a variable that has a non-linear,
    # lower-tail dependency on u1.
    u2_given_u1 = ((u1 ** (-theta)) * (u2 ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)

    # 3) Transform the uniform marginals (u1, u2_given_u1) to standard normal marginals (eps_T, eps_Y)
    # This is achieved using the inverse of the Cumulative Distribution Function (icdf)
    # for a standard normal distribution (mean=0, std=1), also known as Inverse Transform Sampling.
    normal_dist = torch.distributions.Normal(0, 1)
    eps_T = normal_dist.icdf(u1)
    eps_Y = normal_dist.icdf(u2_given_u1)

    # 4) Standardize the resulting noise terms to have a mean of exactly 0 and a std of 1
    # This is a crucial step to ensure a fair comparison with the standard Gaussian case,
    # as it isolates the effect of the dependency structure from changes in the marginals.
    eps_T = (eps_T - eps_T.mean()) / eps_T.std() # check that later do i really need this?
    eps_Y = (eps_Y - eps_Y.mean()) / eps_Y.std()

    # 5) Generate the observable variables T and Y using the Structural Causal Model (SCM) equations
    T_data = eps_T
    Y_data = alpha * T_data + eps_Y

    return T_data, Y_data