from astropy.io import fits
from lmfit import minimize, Parameters
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class PulsatingRVModel:
    def __init__(self, P, E, filepath):
        self.P = P
        self.E = E
        self.filepath = filepath
        self.params = Parameters()

    def fit(self, NFS):
        with fits.open(filepath) as hdul:
            self.data = hdul[1].data
        self.setup_params(NFS)
        return minimize(self.residual, self.params)

    def setup_params(self, NFS):
        self.params.add('vgamma', min=np.min(self.data['RV']), max=np.max(self.data['RV']))
        for n in range(1, NFS+1):
            self.params.add(f'a{n}', min=-150.0, max=150.0)
            self.params.add(f'b{n}', min=-150.0, max=150.0)

    def RV_model(self, params, t):
        phi = (t - self.E)/self.P
        RV = params['vgamma']
        for n in range(1, NFS+1):
            RV += params[f'a{n}']*np.cos(2*n*np.pi*phi) + params[f'b{n}']*np.sin(2*n*np.pi*phi)
        return RV

    def residual(self, params):
        RV = self.RV_model(params, self.data['BJD'])
        return (self.data['RV'] - RV)/self.data['RV_ERR']


P = 14.788141
E = 57540.295599
filepath = 'VELOCE_DR1_FITS/RW_Cas.fits'
NFS = 10

model = PulsatingRVModel(P, E, filepath)
out = model.fit(NFS)

x = np.linspace(E, E + P, num=1000)
y = model.RV_model(out.params, x)
x = np.linspace(0.0, 1.0, num=1000)

plt.scatter((model.data['BJD'] - E)/P % 1, model.data['RV'])
plt.plot(x, y)

plt.show()

plt.scatter((model.data['BJD'] - E)/P % 1, model.data['RV'] - model.RV_model(out.params, model.data['BJD']))

plt.show()
