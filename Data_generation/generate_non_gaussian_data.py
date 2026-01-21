# Generate synthetic data with non-Gaussian noise
def generate_non_gaussian_data(num_samples, alpha, non_linear_strength):
    """
    Generates synthetic data where the underlying noise has a non-linear (quadratic) dependency.
    This function implements the h(eps_T, eta) approach to create a non-Gaussian joint noise distribution.

    Args:
        num_samples (int): The number of data points to generate.
        alpha (float): The true causal effect (ACE) to embed in the data.
        non_linear_strength (float): A coefficient to control the strength of the
                                     non-linear (U-shaped) dependency between noise terms.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated T_data and Y_data tensors,
                                           each with shape [num_samples, 1].
    """
    # 1) Generate two independent standard normal noise sources
    # eps_T will be the noise for T, and eta is an independent noise source for Y
    eps_T = torch.randn(num_samples, 1)
    eta = torch.randn(num_samples, 1)

    # 2) Create a non-linear (quadratic) dependency to define eps_Y.
    # The function h(x) is x**2, creating a U-shaped relationship.
    eps_Y = non_linear_strength  * eps_T**2 + eta

    # 3) Standardize the resulting eps_Y to have a mean of 0 and a standard deviation of 1.
    # This is crucial for a fair comparison with the standard Gaussian case.
    eps_Y = (eps_Y - eps_Y.mean()) / eps_Y.std()

    # 4) Generate the observable variables T and Y using the Structural Causal Model (SCM).
    T_data = eps_T
    Y_data = alpha * T_data + eps_Y

    return T_data, Y_data