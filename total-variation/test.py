import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic import plot_shotrecord
from examples.seismic import demo_model, plot_velocity, plot_perturbation
from examples.seismic.acoustic import AcousticWaveSolver
from devito import Function, clear_cache
#%matplotlib inline

configuration['log_level'] = 'WARNING'

nshots = 9  # Number of shots to create gradient from
nreceivers = 101  # Number of receiver locations per shot 
fwi_iterations = 8  # Number of outer FWI iterations

#NBVAL_IGNORE_OUTPUT

# Define true and initial model
shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('circle-isotropic', vp=3.0, vp_background=2.5,
                    origin=origin, shape=shape, spacing=spacing, nbpml=40)

model0 = demo_model('circle-isotropic', vp=2.5, vp_background=2.5,
                     origin=origin, shape=shape, spacing=spacing, nbpml=40)

#plot_velocity(model)
#plot_velocity(model0)
#plot_perturbation(model0, model)

#NBVAL_IGNORE_OUTPUT
# Define acquisition geometry: source

# Define time discretization according to grid spacing
t0 = 0.
tn = 1000.  # Simulation lasts 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
nt = int(1 + (tn-t0) / dt)  # Discrete time axis length
time = np.linspace(t0, tn, nt)  # Discrete modelling time

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0, time=np.linspace(t0, tn, nt))
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, 0] = 20.  # 20m from the left end

# We can plot the time signature to see the wavelet
#src.show()

#NBVAL_IGNORE_OUTPUT
# Define acquisition geometry: receivers

# Initialize receivers for synthetic data
rec = Receiver(name='rec', grid=model.grid, npoint=nreceivers, ntime=nt)
rec.coordinates.data[:, 1] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec.coordinates.data[:, 0] = 980. # 20m from the right end

# Plot acquisition geometry
#plot_velocity(model, source=src.coordinates.data, receiver=rec.coordinates.data[::4, :])


# Compute synthetic data with forward operator 

solver = AcousticWaveSolver(model, src, rec, space_order=4)
true_d, _, _ = solver.forward(src=src, m=model.m)

# Compute initial data with forward operator 
smooth_d, _, _ = solver.forward(src=src, m=model0.m)

#NBVAL_IGNORE_OUTPUT

# Plot shot record for true and smooth velocity model and the difference
#plot_shotrecord(true_d.data, model, t0, tn)
#plot_shotrecord(smooth_d.data, model, t0, tn)
#plot_shotrecord(smooth_d.data - true_d.data, model, t0, tn)
#plot_shotrecord(smooth_d.data - smooth_d.data, model, t0, tn)


#print(src.coordinates.data[0,:])
#print(model.domain_size)


#NBVAL_IGNORE_OUTPUT

# Prepare the varying source locations sources
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = 30.
source_locations[:, 1] = np.linspace(0., 1000, num=nshots)

#plot_velocity(model, source=source_locations)

# Create FWI gradient kernel 

def fwi_gradient(m_in):
    # Important: We force previous wavefields to be destroyed,
    # so that we may reuse the memory.
    clear_cache()
    
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid,
                        ntime=nt, coordinates=rec.coordinates.data)
    objective = 0.
    
    for i in range(nshots):
        # Update source location
        src.coordinates.data[0, :] = source_locations[i, :]
        
        # Generate synthetic data from true model
        true_d, _, _ = solver.forward(src=src, m=model.m)
        
        # Compute smooth data and full forward wavefield u0
        smooth_d, u0, _ = solver.forward(src=src, m=m_in, save=True)
        
        # Compute gradient from data residual and update objective function 
        residual.data[:] = smooth_d.data[:] - true_d.data[:]
        objective += .5*np.linalg.norm(residual.data.reshape(-1))**2
        solver.gradient(rec=residual, u=u0, m=m_in, grad=grad)
    
    return objective, grad.data

# Compute gradient of initial model
ff, update = fwi_gradient(model0.m)
print('Objective value is %f ' % ff)
print(src.coordinates.data[0,:])
print(source_locations[0, :])
print(source_locations[1, :])
print(update)

