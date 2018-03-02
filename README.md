# MF3D
Line searching interferometric data cubes. Developed for the COLDz VLA project, See Pavesi et al., 2018a, submitted, for a description.

Basics:
To start, you need to produce a signal-to-noise cube, i.e., you have to normalize your data cube so that the noise is everywhere equal to one. If your cube just came out of CLEAN, and the noise is spatially uniform, this may just require going through your cube channel by channel and dividing by the standard deviation of each channel (if your noise is spatially varying, you can read my paper for techniques to deal with it).

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

Create the MF3D object giving it the SNR cube, the spatial template sizes, and channel template widths (see below for details).

	MF_c=MF3D(data,[0,3,6,9,12,15],[4,8,12,16,20])


Choosing template sizes:
The templates are circular 2D gaussians, and frequency 1D gaussians. Sizes are provided as FWHM, in pixels and channels, respectively.
Template matching is achieved when the frequency width is the same as the source line width. For the spatial size, things are different because of the correlated nature of interferometric noise (i.e., the beam). Point-like templates (size=0) are required to select unresolved sources, larger templates select slightly extended sources of increasing size. The relation between template spatial size and selected source size is complicated and best assessed by running simulations with fake injected sources. An approximate matching formula is given by $\sigma_A^2$=$\sigma_h^2+2\sigma_b^2$, where $\sigma_A$ is the convolved size (i.e., observed) of the source, $\sigma_h$ is the template size and $\sigma_b$ is the beam size. We recommend always including the 0-size template, to capture point-sources.

More techincal details:
The MF3D class may need to be modified in different ways. For example, the integer variable "num_blocks" determined how many blocks the cube is split into (along each axis) when taking the FFT, it's fastest to keep it small, but large cubes may need more splits to keep the memory usage within the available RAM.
Another possible change may be to the freq_rang, and spat_rang variables in the make_combined_(positives, negatives) functions. These are the sizes of the small cubes used to find local peaks of the matched filtered cubes. It's unlikely to make a large difference, but be careful. Make sure these parameters match for positive and negatives!

Other hyper-parameters that may need tuning are the 

	return (1.*a-1.*d)**2+(1.*b-1.*e)**2<25. and np.absolute(1.*c-1.*f)<15.
	
conditions inside the make_reduint_(positive, negative) functions. These are the conditions for identifying candidates selected in different templates, as belonging to the same line feature. The spatial part is set to a radial separation of 5 pixels, and the channel separation corresponds to 15 channels. These may need tuning for your specific cases, and more complex criteria may be attempted as well using different thresholds for different templates. We did not find significant differences of any importance, but be aware of these parameters. Again, make sure these parameters match for positive and negatives!

Output:
Mostly, you'll care about these files: working/reduint_negative.dat, working/reduint_positive.dat
These are cPickle files containing a python list, each. Read them back with:

	import cPickle
	inp=open('working/reduint_positive.dat')
	positive_features=cPickle.load(inp)
	inp.close()

The list contains all features down to a SNR=4, each feature is a tuple with this structure:
	
	(SNR, (x_pos,y_pos,chan_pos), Ntempl_detect, (peak_spat,peak_freq))

Where the position in the cube is given in pixels and channel, Ntempl_detect is the number of templates for which this feature is detected above SNR=4, and the peak_spat and peak_freq contain the sizes of the template that gives the highest SNR (reported), i.e., the best-matching template.

Good luck!
