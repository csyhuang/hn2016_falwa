import os
import numpy as np
import matplotlib.pyplot as plt
from falwa.oopinterface import QGFieldNHN22
from falwa.constant import SCALE_HEIGHT, P_GROUND

load_path = os.getcwd() + '/npy_file.npy'
numpy_dicti = np.load('./npy_file.npy', allow_pickle=True)

def file(key='uu', numpy_dicti=numpy_dicti):
    return numpy_dicti.item().get(key)

qgfield_object = QGFieldNHN22(
    file('xlon'), file('ylat'), file('plev'), file('uu'), file('vv'), file('tt'),
    kmax=81, dz=500, eq_boundary_index=file('eq_boundary_index'))
equator_idx = qgfield_object.equator_idx

hlev = -SCALE_HEIGHT*np.log(file('plev')/P_GROUND)
plt.contourf(file('ylat'), hlev, file('uu').mean(axis=-1), 40, cmap="rainbow")
plt.title('zonal mean u before interpolation')
plt.xlabel('latitude[deg]')
plt.ylabel('pseudoheight[m]')
plt.colorbar()
plt.savefig("zonal_mean_u_b4_interpolation.png")
plt.show()


qgfield_object.interpolate_fields(return_named_tuple=False)
qgfield_object.compute_reference_states(return_named_tuple=False)
xx = qgfield_object.qref

plt.contourf(file('ylat'), qgfield_object.height, qgfield_object.interpolated_u.mean(axis=-1), 40, cmap="rainbow")
plt.title('zonal mean u after interpolation')
plt.xlabel('latitude[deg]')
plt.ylabel('pseudoheight[m]')
plt.colorbar()
plt.savefig("zonal_mean_u_after_interpolation.png")
plt.show()

plt.contourf(file('ylat'), qgfield_object.height, qgfield_object.qgpv.mean(axis=-1), 40, cmap="rainbow")
plt.title('zonal mean qgpv')
plt.xlabel('latitude[deg]')
plt.ylabel('pseudoheight[m]')
plt.colorbar()
plt.savefig("zonal_mean_qgpv.png")
plt.show()

plt.contourf(file('ylat'), qgfield_object.height, qgfield_object.qref, 40, cmap="rainbow")
plt.title('qref')
plt.xlabel('latitude[deg]')
plt.ylabel('pseudoheight[m]')
plt.colorbar()
plt.savefig("qref.png")
plt.show()

plt.contourf(file('ylat'), qgfield_object.height, qgfield_object.uref, 40, cmap="rainbow")
plt.title('uref')
plt.xlabel('latitude[deg]')
plt.ylabel('pseudoheight[m]')
plt.colorbar()
plt.savefig("uref.png")
plt.show()
print("Yes")
# qgfield_object.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)