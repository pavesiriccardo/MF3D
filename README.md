# MF3D
Line searching interferometric data cubes

To start you need to produce a signal-to-noise cube, i.e., you have to normalize your data cube so that the noise is everywhere equal to one. If your cube just came out of CLEAN, and the noise is spatially uniform, this may just require going through your cube channel by channel and dividing by the standard deviation of each channel (if your noise is spatially varying, you can read my paper for techniques to deal with it).

	f=pyfits.open(fits_file_name)
	data=f[0].data     #If you have a degenerate first axis, use  f[0].section[0] instead
	stdd=np.nanstd(data,(1,2))   #This is an array with the noise vector, one per channel. Make sure you don't have zeros in the channel maps, and replace them with nan instead.

	for chan in range(Nchan):
		data[chan]/=stdd[chan]

"data" now contains the signal-to-noise cube (numpy array, of course)

MAKE SURE the noise in your channels is NOT correlated! So, calculate the following, and check it's within a few percent of 1. If you have correlation, you may have to re-image with interpolation='nearest' instead.

	corr=[np.nanstd(data[chan]+data[chan+1])/np.sqrt(np.nanstd(data[chan])**2+np.nanstd(data[chan+1])**2) for chan in range(Nchan-1)]  

Import the MF3D class with something like: 

	from MF3D_class import MF3D

Create the MF3D object giving it the SNR cube, and the spatial template sizes, and channel template widths.

	MF_c=MF3D(data,[0,3,6,9,12,15],[4,8,12,16,20])



