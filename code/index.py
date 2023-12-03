from random import gauss, uniform
import numpy
import os
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt




import os

figures_directory = '../figures'
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)

# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

# DATA_ROOT = os.path.join('..', 'data')


# DATA_ROOT = os.path.join('..', 'data')
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

PATH_TO_INPUTS = os.path.join(DATA_ROOT, 'inputs.csv')
PATH_TO_2D_1 = os.path.join(DATA_ROOT, 'points_2d_camera_1.csv')
PATH_TO_2D_2 = os.path.join(DATA_ROOT, 'points_2d_camera_2.csv')

# FIGURES_ROOT = os.path.join('..', 'figures')

# figures_directory = os.path.abspath('../figures')
# plt.savefig(os.path.join(figures_directory, f"{filename_base}.png"), format='png')
FIGURES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))


CAMERA_1 = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
CAMERA_2 = numpy.array([[0,0,1,-5],[0,1,0,0],[-1,0,0,5]])

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def calcLogLikelihood(M, rs, pi, pf, cov, ts):
    # Create normalized pi and pf vectors for matrix multiplication
    normalized_pi = numpy.zeros(4)
    normalized_pf = numpy.zeros(4)
    for i in range(len(pi)):
        normalized_pi[i] = pi[i]
        normalized_pf[i] = pf[i]
    
    # Add the normalizaiton param
    normalized_pi[3] = 1
    normalized_pf[3] = 1

    # Follow generative process to find u*, v*, w*
    uvw_qi = numpy.dot(M, normalized_pi)
    uvw_qf = numpy.dot(M, normalized_pf)

    # Find qi, qf from u*, v*, w*
    qi = 1/uvw_qi[2] * numpy.array([uvw_qi[0], uvw_qi[1]])
    qf = 1/uvw_qf[2] * numpy.array([uvw_qf[0], uvw_qf[1]])
    
    # Declare point variable (to find through the generative process and linear interpolation with the inputs t)
    qs = []

    # Declare response variable
    log_likelihood = 0

    # Find each q point from the generative process and the linear interpolation from inputs
    for i in range(len(ts)):
        qs.append(qi + (qf - qi) * ts[i])
    
    # Calculate log likelihood for each point and sum all of them together
    for i in range(len(qs)):
        log_likelihood += multivariate_normal.logpdf(rs[i], qs[i], cov)
    
    return log_likelihood

def calcLogLikelihoodT5(M1, M2, rs_1, rs_2, pi, pf, cov, ts):
    # Create normalized pi and pf vectors for matrix multiplication
    normalized_pi = numpy.zeros(4)
    normalized_pf = numpy.zeros(4)
    for i in range(len(pi)):
        normalized_pi[i] = pi[i]
        normalized_pf[i] = pf[i]
    
    # Add the normalizaiton param
    normalized_pi[3] = 1
    normalized_pf[3] = 1

    # Follow generative process to find u*, v*, w* for camera 1
    uvw_qi_cam1 = numpy.dot(M1, normalized_pi)
    uvw_qf_cam1 = numpy.dot(M1, normalized_pf)
    
    # Follow generative process to find u*, v*, w* for camera 1
    uvw_qi_cam2 = numpy.dot(M2, normalized_pi)
    uvw_qf_cam2 = numpy.dot(M2, normalized_pf)


    # Find qi, qf from u*, v*, w* for camera 1
    qi_cam1 = 1/uvw_qi_cam1[2] * numpy.array([uvw_qi_cam1[0], uvw_qi_cam1[1]])
    qf_cam1 = 1/uvw_qf_cam1[2] * numpy.array([uvw_qf_cam1[0], uvw_qf_cam1[1]])
    
    # Find qi, qf from u*, v*, w* for camera 2
    qi_cam2 = 1/uvw_qi_cam2[2] * numpy.array([uvw_qi_cam2[0], uvw_qi_cam2[1]])
    qf_cam2 = 1/uvw_qf_cam2[2] * numpy.array([uvw_qf_cam2[0], uvw_qf_cam2[1]])
    
    # Declare point variable (to find through the generative process and linear interpolation with the inputs t)
    # for both cameras
    qs_cam1, qs_cam2 = [], []

    # Declare response variable
    log_likelihood = 0

    # Find each q point from the generative process and the linear interpolation from inputs
    for i in range(len(ts)):
        qs_cam1.append(qi_cam1 + (qf_cam1 - qi_cam1) * ts[i])
        qs_cam2.append(qi_cam2 + (qf_cam2 - qi_cam2) * ts[i])
    
    # Calculate log likelihood for each point and sum all of them together
    for i in range(len(qs_cam1)):
        log_likelihood += multivariate_normal.logpdf(rs_1[i], qs_cam1[i], cov)
        log_likelihood += multivariate_normal.logpdf(rs_2[i], qs_cam2[i], cov)
    
    return log_likelihood


# -----------------------------------------------------------------------------
# Main Method
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Load inputs t_1, ..., t_s
    inputs = numpy.loadtxt(PATH_TO_INPUTS)

    # Load 2D points r_1, ..., r_s (the observed points in the camera u,v image plane)
    points2d = numpy.loadtxt(PATH_TO_2D_1, delimiter=',')
    
    # Load 2D points r_1, ..., r_s (the observed points in the camera u,v image plane)
    points2d_cam2 = numpy.loadtxt(PATH_TO_2D_2, delimiter=',')

    # -----------------------------------------------------------------------------
    # TASK 1
    # -----------------------------------------------------------------------------
    # Calculate prior distribution
    prior_mean = numpy.array([0,0,4])
    prior_covariance = 6 * numpy.identity(3)
    prior_distribution = multivariate_normal(prior_mean, prior_covariance)

    # Sampling for p_i and p_f
    initial_pi, initial_pf = prior_distribution.rvs(2)

    # To define p(w_s* | w_s-1), we'll propose a density. In other words, p(w_s* | w_s-1, sigma) = N(w_s-1, sigma)
    # Density does not have to have any connection with the posterior, so we'll pick a covariance
    proposed_covariance = 0.5

    # Covariance for gaussian noise
    gaussian_cov = 0.05**2 * numpy.identity(2)

    N = 50 * 1000   # 50,000 samples
    # N = 100

    # Accepted proposals
    accepted_proposals_pi = []
    accepted_proposals_pf = []
    
    # Accepted proposals's posteriors
    accepted_proposals_pi_posterior = []
    accepted_proposals_pf_posterior = []
    
    # Metropolis-Hastings starts, compute N samples
    for i in range(N):
        # Sample two proposed new values for p_i and p_f
        proposed_pi = multivariate_normal(initial_pi, proposed_covariance).rvs()
        proposed_pf = multivariate_normal(initial_pf, proposed_covariance).rvs()

        # Calculate joint probability density functions for both initial values and proposed values, 
        # to use for finding the posterior (priors)
        initial_joint_pdf = prior_distribution.logpdf(initial_pi) + prior_distribution.logpdf(initial_pf)
        proposed_joint_pdf = prior_distribution.logpdf(proposed_pi) + prior_distribution.logpdf(proposed_pf)

        # Calculate log likelihood for both initial values and proposed values (likelihood)
        initial_log_likelihood = calcLogLikelihood(CAMERA_1, points2d, initial_pi, initial_pf, gaussian_cov, inputs)
        proposed_log_likelihood = calcLogLikelihood(CAMERA_1, points2d, proposed_pi, proposed_pf, gaussian_cov, inputs)

        # Calculate posteriors from numerator of posterior equation (likelihood * prior = logLikelihood + logPrior)
        initial_posterior = initial_joint_pdf + initial_log_likelihood
        proposed_posterior = proposed_joint_pdf + proposed_log_likelihood

        # Calculate ratio of posterior densities in log space and then convert it back to normal space.
        # From r = p(w_s* | sig^2) / p(w_s-1 | sig^2)
        r = numpy.exp(proposed_posterior - initial_posterior)

        # If ratio is less than one, proposal may still be accepted based on a random draw from a uniform distribution
        if r < 1:
            # Make a random draw from a uniform distribution
            u = uniform(0,1)

            # If u <= r, accept the proposal
            if u <= r:
                initial_pi, initial_pf = proposed_pi, proposed_pf
        # Else if ratio is more than one, proposal is always accepted
        else:
            initial_pi, initial_pf = proposed_pi, proposed_pf

        accepted_proposals_pi.append(initial_pi)
        accepted_proposals_pf.append(initial_pf)

        if i == N * 1/10:
            print('M-H is 0.1 done')
        elif i == N * 2/10:
            print('M-H is 0.2 done')
        elif i == N * 3/10:
            print('M-H is 0.3 done')
        elif i == N * 4/10:
            print('M-H is 0.4 done')
        elif i == N * 5/10:
            print('M-H is 0.5 done')
        elif i == N * 6/10:
            print('M-H is 0.6 done')
        elif i == N * 7/10:
            print('M-H is 0.7 done')
        elif i == N * 8/10:
            print('M-H is 0.8 done')
        elif i == N * 9/10:
            print('M-H is 0.9 done')
    
    # -----------------------------------------------------------------------------
    # TASK 2
    # -----------------------------------------------------------------------------
    accepted_x_pi, accepted_y_pi, accepted_z_pi = [], [], []
    for proposal in accepted_proposals_pi:
        accepted_x_pi.append(proposal[0])
        accepted_y_pi.append(proposal[1])
        accepted_z_pi.append(proposal[2])
    
    accepted_x_pf, accepted_y_pf, accepted_z_pf = [], [], []
    for proposal in accepted_proposals_pf:
        accepted_x_pf.append(proposal[0])
        accepted_y_pf.append(proposal[1])
        accepted_z_pf.append(proposal[2])
    
    plt.figure()
    plt.plot(accepted_x_pi, label='$p_i$ (x)')
    plt.plot(accepted_y_pi, label='$p_i$ (y)')
    plt.plot(accepted_z_pi, label='$p_i$ (z)')
    plt.xlabel('Samples')
    plt.ylabel('$p_i$')
    plt.legend(loc="upper left")
    plt.title('Accepted proposals for $p_i$')
    filename_base = 'accepted_pi'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()
    
    plt.figure()
    plt.plot(accepted_x_pf, label='$p_f$ (x)')
    plt.plot(accepted_y_pf, label='$p_f$ (y)')
    plt.plot(accepted_z_pf, label='$p_f$ (z)')
    plt.xlabel('Samples')
    plt.ylabel('$p_f$')
    plt.legend(loc="upper left")
    plt.title('Accepted proposals for $p_f$')
    filename_base = 'accepted_pf'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()

    # -----------------------------------------------------------------------------
    # TASK 3
    # -----------------------------------------------------------------------------
    pi_MAP = numpy.array([numpy.mean(accepted_x_pi), numpy.mean(accepted_y_pi), numpy.mean(accepted_z_pi)])
    pf_MAP = numpy.array([numpy.mean(accepted_x_pf), numpy.mean(accepted_y_pf), numpy.mean(accepted_z_pf)])

    # Create normalized pi and pf vectors for matrix multiplication
    normalized_pi_MAP = numpy.zeros(4)
    normalized_pf_MAP = numpy.zeros(4)
    for i in range(len(pi_MAP)):
        normalized_pi_MAP[i] = pi_MAP[i]
        normalized_pf_MAP[i] = pf_MAP[i]

    normalized_pi_MAP[3] = 1
    normalized_pf_MAP[3] = 1

    # Follow generative process to find u*, v*, w*
    uvw_qi_MAP = numpy.dot(CAMERA_1, normalized_pi_MAP)
    uvw_qf_MAP = numpy.dot(CAMERA_1, normalized_pf_MAP)

    # Find qi, qf in Camera 1's perspective from u*, v*, w*
    qi_MAP = 1/uvw_qi_MAP[2] * numpy.array([uvw_qi_MAP[0], uvw_qi_MAP[1]])
    qf_MAP = 1/uvw_qf_MAP[2] * numpy.array([uvw_qf_MAP[0], uvw_qf_MAP[1]])

    plt.figure()
    plt.plot(qi_MAP, qf_MAP, 'bo-', label='True line')
    plt.plot([ points2d[i][0] for i in range(len(points2d)) ], [ points2d[i][1] for i in range(len(points2d)) ], 'ro', label='Original Noisy Observations')
    plt.xlabel('$v$')
    plt.ylabel('$u$')
    plt.legend(loc="upper left")
    plt.title('MAP Estimate Camera 1')
    filename_base = 'map_cam1'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()

    # -----------------------------------------------------------------------------
    # TASK 4
    # -----------------------------------------------------------------------------
    # Create normalized pi and pf vectors for matrix multiplication
    normalized_pi_MAP_cam2 = numpy.zeros(4)
    normalized_pf_MAP_cam2 = numpy.zeros(4)
    for i in range(len(pi_MAP)):
        normalized_pi_MAP_cam2[i] = pi_MAP[i]
        normalized_pf_MAP_cam2[i] = pf_MAP[i]

    normalized_pi_MAP_cam2[3] = 1
    normalized_pf_MAP_cam2[3] = 1

    # Follow generative process to find u*, v*, w*
    uvw_qi_MAP_cam2 = numpy.dot(CAMERA_2, normalized_pi_MAP_cam2)
    uvw_qf_MAP_cam2 = numpy.dot(CAMERA_2, normalized_pf_MAP_cam2)

    # Find qi, qf in Camera 1's perspective from u*, v*, w*
    qi_MAP_cam2 = 1/uvw_qi_MAP_cam2[2] * numpy.array([uvw_qi_MAP_cam2[0], uvw_qi_MAP_cam2[1]])
    qf_MAP_cam2 = 1/uvw_qf_MAP_cam2[2] * numpy.array([uvw_qf_MAP_cam2[0], uvw_qf_MAP_cam2[1]])

    plt.figure()
    plt.plot(qi_MAP_cam2, qf_MAP_cam2, 'bo-', label='True line')
    plt.plot([ points2d_cam2[i][0] for i in range(len(points2d_cam2)) ], [ points2d_cam2[i][1] for i in range(len(points2d_cam2)) ], 'ro', label='Original Noisy Observations (Camera 2)')
    plt.xlabel('$v$')
    plt.ylabel('$u$')
    plt.legend(loc="upper left")
    plt.title('MAP Estimate Camera 2')
    filename_base = 'map_cam2'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()

    # -----------------------------------------------------------------------------
    # TASK 5
    # -----------------------------------------------------------------------------
    accepted_proposals_pi = []
    accepted_proposals_pf = []
    # Metropolis-Hastings starts, compute N samples
    for i in range(N):
        # Sample two proposed new values for p_i and p_f
        proposed_pi = multivariate_normal(initial_pi, proposed_covariance).rvs()
        proposed_pf = multivariate_normal(initial_pf, proposed_covariance).rvs()

        # Calculate joint probability density functions for both initial values and proposed values, 
        # to use for finding the posterior (priors)
        initial_joint_pdf = prior_distribution.logpdf(initial_pi) + prior_distribution.logpdf(initial_pf)
        proposed_joint_pdf = prior_distribution.logpdf(proposed_pi) + prior_distribution.logpdf(proposed_pf)

        # Calculate log likelihood for both initial values and proposed values (likelihood)
        initial_log_likelihood = calcLogLikelihoodT5(CAMERA_1, CAMERA_2, points2d, points2d_cam2, initial_pi, initial_pf, gaussian_cov, inputs)
        proposed_log_likelihood = calcLogLikelihoodT5(CAMERA_1, CAMERA_2, points2d, points2d_cam2, proposed_pi, proposed_pf, gaussian_cov, inputs)

        # Calculate posteriors from numerator of posterior equation (likelihood * prior = logLikelihood + logPrior)
        initial_posterior = initial_joint_pdf + initial_log_likelihood
        proposed_posterior = proposed_joint_pdf + proposed_log_likelihood

        # Calculate ratio of posterior densities in log space and then convert it back to normal space.
        # From r = p(w_s* | sig^2) / p(w_s-1 | sig^2)
        r = numpy.exp(proposed_posterior - initial_posterior)

        # If ratio is less than one, proposal may still be accepted based on a random draw from a uniform distribution
        if r < 1:
            # Make a random draw from a uniform distribution
            u = uniform(0,1)

            # If u <= r, accept the proposal
            if u <= r:
                initial_pi, initial_pf = proposed_pi, proposed_pf
        # Else if ratio is more than one, proposal is always accepted
        else:
            initial_pi, initial_pf = proposed_pi, proposed_pf

        accepted_proposals_pi.append(initial_pi)
        accepted_proposals_pf.append(initial_pf)

        if i == N * 1/10:
            print('M-H T5 is 0.1 done')
        elif i == N * 2/10:
            print('M-H T5 is 0.2 done')
        elif i == N * 3/10:
            print('M-H T5 is 0.3 done')
        elif i == N * 4/10:
            print('M-H T5 is 0.4 done')
        elif i == N * 5/10:
            print('M-H T5 is 0.5 done')
        elif i == N * 6/10:
            print('M-H T5 is 0.6 done')
        elif i == N * 7/10:
            print('M-H T5 is 0.7 done')
        elif i == N * 8/10:
            print('M-H T5 is 0.8 done')
        elif i == N * 9/10:
            print('M-H T5 is 0.9 done')

    accepted_x_pi, accepted_y_pi, accepted_z_pi = [], [], []
    for proposal in accepted_proposals_pi:
        accepted_x_pi.append(proposal[0])
        accepted_y_pi.append(proposal[1])
        accepted_z_pi.append(proposal[2])
    
    accepted_x_pf, accepted_y_pf, accepted_z_pf = [], [], []
    for proposal in accepted_proposals_pf:
        accepted_x_pf.append(proposal[0])
        accepted_y_pf.append(proposal[1])
        accepted_z_pf.append(proposal[2])
    
    plt.figure()
    plt.plot(accepted_x_pi, label='$p_i$ (x)')
    plt.plot(accepted_y_pi, label='$p_i$ (y)')
    plt.plot(accepted_z_pi, label='$p_i$ (z)')
    plt.xlabel('Samples')
    plt.ylabel('$p_i$')
    plt.legend(loc="upper left")
    plt.title('Accepted proposals for $p_i$ (both cams)')
    filename_base = 'accepted_pi_both_cams'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()
    
    plt.figure()
    plt.plot(accepted_x_pf, label='$p_f$ (x)')
    plt.plot(accepted_y_pf, label='$p_f$ (y)')
    plt.plot(accepted_z_pf, label='$p_f$ (z)')
    plt.xlabel('Samples')
    plt.ylabel('$p_f$')
    plt.legend(loc="upper left")
    plt.title('Accepted proposals for $p_f$ (both cams)')
    filename_base = 'accepted_pf_both_cams'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()

    pi_MAP = numpy.array([numpy.mean(accepted_x_pi), numpy.mean(accepted_y_pi), numpy.mean(accepted_z_pi)])
    pf_MAP = numpy.array([numpy.mean(accepted_x_pf), numpy.mean(accepted_y_pf), numpy.mean(accepted_z_pf)])

    # Create normalized pi and pf vectors for matrix multiplication
    normalized_pi_MAP = numpy.zeros(4)
    normalized_pf_MAP = numpy.zeros(4)
    for i in range(len(pi_MAP)):
        normalized_pi_MAP[i] = pi_MAP[i]
        normalized_pf_MAP[i] = pf_MAP[i]

    normalized_pi_MAP[3] = 1
    normalized_pf_MAP[3] = 1

    # Follow generative process to find u*, v*, w*
    uvw_qi_MAP = numpy.dot(CAMERA_1, normalized_pi_MAP)
    uvw_qf_MAP = numpy.dot(CAMERA_1, normalized_pf_MAP)

    # Find qi, qf in Camera 1's perspective from u*, v*, w*
    qi_MAP = 1/uvw_qi_MAP[2] * numpy.array([uvw_qi_MAP[0], uvw_qi_MAP[1]])
    qf_MAP = 1/uvw_qf_MAP[2] * numpy.array([uvw_qf_MAP[0], uvw_qf_MAP[1]])

    plt.figure()
    plt.plot(qi_MAP, qf_MAP, 'bo-', label='True line')
    plt.plot([ points2d[i][0] for i in range(len(points2d)) ], [ points2d[i][1] for i in range(len(points2d)) ], 'ro', label='Original Noisy Observations')
    plt.xlabel('$v$')
    plt.ylabel('$u$')
    plt.legend(loc="upper left")
    plt.title('MAP Estimate Camera 1 (both cams)')
    filename_base = 'map_cam1_both_cams'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()

    # Follow generative process to find u*, v*, w*
    uvw_qi_MAP = numpy.dot(CAMERA_2, normalized_pi_MAP)
    uvw_qf_MAP = numpy.dot(CAMERA_2, normalized_pf_MAP)

    # Find qi, qf in Camera 2's perspective from u*, v*, w*
    qi_MAP = 1/uvw_qi_MAP[2] * numpy.array([uvw_qi_MAP[0], uvw_qi_MAP[1]])
    qf_MAP = 1/uvw_qf_MAP[2] * numpy.array([uvw_qf_MAP[0], uvw_qf_MAP[1]])

    plt.figure()
    plt.plot(qi_MAP, qf_MAP, 'bo-', label='True line')
    plt.plot([ points2d_cam2[i][0] for i in range(len(points2d_cam2)) ], [ points2d_cam2[i][1] for i in range(len(points2d_cam2)) ], 'ro', label='Original Noisy Observations')
    plt.xlabel('$v$')
    plt.ylabel('$u$')
    plt.legend(loc="upper left")
    plt.title('MAP Estimate Camera 2 (both cams)')
    filename_base = 'map_cam2_both_cams'
    plt.savefig(f'../figures/{filename_base}.png', format='png')
    plt.show()