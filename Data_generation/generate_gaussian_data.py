# Generate synthetic data with Gaussian noise
def generate_gaussian_data(num_samples, alpha, rho):
    """
    Generates synthetic data where the underlying noise has a linear dependency
    structure defined by a Gaussian copula (Multivariate Normal distribution).

    Args:
        num_samples (int): The number of data points to generate.
        alpha (float): The true causal effect (ACE) to embed in the data.
        rho (float): The confounding parameter, representing the Pearson correlation
                     between the noise terms (eps_T and eps_Y).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated T_data and Y_data tensors,
                                           each with shape [num_samples, 1].
    """
    # 1) Define the properties of the bivariate Gaussian noise distribution.
    # The mean vector is [0, 0].
    mean_vector = torch.zeros(2)
    # The covariance matrix encodes the linear dependency (confounding) via rho
    covariance_matrix = torch.tensor([[1.0, rho], [rho, 1.0]])
    # Create the distribution object that will act as our noise generator.
    noise_dist = MultivariateNormal(mean_vector, covariance_matrix)

    # 2) Sample noise pairs of (eps_T, eps_Y) from the defined distribution.
    noise_samples = noise_dist.sample((num_samples,))
    # Reshape noise vectors to be column vectors [num_samples, 1].
    eps_T = noise_samples[:, 0].unsqueeze(1)
    eps_Y = noise_samples[:, 1].unsqueeze(1)

    # 3) Generate the observable variables T and Y using the Structural Causal Model (SCM).
    T_data = eps_T
    Y_data = alpha * T_data + eps_Y

    return T_data, Y_data