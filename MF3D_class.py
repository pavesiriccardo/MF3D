"""
Copyright (C) 2018, Riccardo Pavesi
E-mail: pavesiriccardo3 -at- gmail.com

Updated versions of the software are available through github:
https://github.com/pavesiriccardo/MF3D
 
If you have found this software useful for your research,
I would appreciate an acknowledgment to the use of the
"Matched Filtering in 3D for interferometry (MF3D) routines of Pavesi et al., (2018b)".
[https://arxiv.org/abs/1808.04372]

This software is provided as is without any warranty whatsoever.
For details of permissions granted please see LICENCE.md
"""
import gc
import numpy as np
import operator, pickle
import os

class MF3D(object):
	def dist3d(self,pos1,pos2):
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)
	def dist2d(self,pos1,pos2):
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
	def __init__(self,SNR,spatial_fwhms,freq_fwhms,working_directory="working/"):   
		"""

		This function initializes the class by loading the S/N data and the list of template sizes to use.

		Parameters
		----------
		SNR : np.ndarray of double
			The S/N data cube, no degenerate axes allowed.

		spatial_fwhms : np.ndarray of double
			Spatial size (as FWHM) of the circular 2D Gaussian template, in pixels.

		freq_fwhms : np.ndarray of double
			Frequency size (as FWHM) of the spectral 1D Gaussian template, in channels.

		Returns
		-------

		"""
		self.SNR=SNR
		if len(self.SNR.shape)>3:
			print('More than 3 dimensions! Need to drop degenerate axes before giving me SNR cube')
		self.Nchan,self.Nypix,self.Nxpix=self.SNR.shape
		if self.Nypix!=self.Nxpix:
			print('So far can only use square images, please fix your cube to spatially square')
		self.SNR=np.where(np.isnan(self.SNR),0,self.SNR)
		self.SNR=np.where(self.SNR==1.,0,self.SNR)
		self.spatial_fwhms=spatial_fwhms
		self.freq_fwhms=freq_fwhms
		self.working_directory=working_directory
	def make_FTSNR(self):   
		"""

		    This function calculates the complex conjugate of the FT of the SNR. We need that because the convolution requires IFT(product) which is equivalent to FT(prod_conjugate)_conjugate/Npix, the template and the final are real so those are not affected by conjugate.
		Creates "working" folder and uses memmaps to handle large data volumes. The output is stored in "working/FTSNR_conjugated".

		    Parameters
		    ----------

		    Returns
		    -------

		"""
		if not os.path.isdir(self.working_directory):
			os.mkdir(self.working_directory)
		ax0=np.memmap(self.working_directory + 'ax0',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		#ax0[:]=np.fft.fftn(self.SNR,axes=[0])
		self.my_fft(self.SNR,ax0,0)
		ax01=np.memmap(self.working_directory + 'ax01',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		#ax01[:]=np.fft.fftn(ax0,axes=[1])
		self.my_fft(ax0,ax01,1)
		del ax0
		os.system('rm ' + self.working_directory + 'ax0')
		ax012=np.memmap(self.working_directory + 'ax012',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		#ax012[:]=np.fft.fftn(ax01,axes=[2])
		self.my_fft(ax01,ax012,2)
		del ax01
		os.system('rm ' + self.working_directory + 'ax01')
		FTt=np.memmap(self.working_directory + 'FTSNR_conjugated',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		FTt[:]=np.conjugate(ax012)
		os.system('rm ' + self.working_directory + 'ax012')
		del ax012
	def my_fft(self,inp_arr,outp_arr,axis):
		"""

		This function calculates the FFT of a large data cube, along a single axis. It splits the axis length in num_blocks blocks, in order to only load a manageble data volume into the RAM.

		Parameters
		----------
		inp_arr: np.ndarray of double
			The array to be FFT'ed, typically will be a memmap for handling large data volumes
		outp_arr: np.ndarray of double
			The output array, typically will be a memmap for handling large data volumes
		axis: int
			Axis index to be FFT'ed

		Returns
		-------

		"""
		num_blocks=2        #How many blocks (per axis) do we want to split the cube into, to reduce memory usage. Keep it as low as your RAM allows you to (2 or 3, perhaps)
		def complete_ends(index,tot_length,error):
			if tot_length-index<error:
				return tot_length
			else:
				return index
		axis0_split=int(inp_arr.shape[0]/num_blocks)
		axis1_split=int(inp_arr.shape[1]/num_blocks)
		axis2_split=int(inp_arr.shape[2]/num_blocks)
		for xblock in range(num_blocks):
			for yblock in range(num_blocks):
				if axis==1:
					outp_arr[xblock*axis0_split:complete_ends((xblock+1)*axis0_split,inp_arr.shape[0],num_blocks),:,yblock*axis2_split:complete_ends((yblock+1)*axis2_split,inp_arr.shape[2],num_blocks)]=np.fft.fft(inp_arr[xblock*axis0_split:complete_ends((xblock+1)*axis0_split,inp_arr.shape[0],num_blocks),:,yblock*axis2_split:complete_ends((yblock+1)*axis2_split,inp_arr.shape[2],num_blocks)],axis=1)
				if axis==0:
					outp_arr[:,xblock*axis1_split:complete_ends((xblock+1)*axis1_split,inp_arr.shape[1],num_blocks),yblock*axis2_split:complete_ends((yblock+1)*axis2_split,inp_arr.shape[2],num_blocks)]=np.fft.fft(inp_arr[:,xblock*axis1_split:complete_ends((xblock+1)*axis1_split,inp_arr.shape[1],num_blocks),yblock*axis2_split:complete_ends((yblock+1)*axis2_split,inp_arr.shape[2],num_blocks)],axis=0)
				if axis==2:
					outp_arr[xblock*axis0_split:complete_ends((xblock+1)*axis0_split,inp_arr.shape[0],num_blocks),yblock*axis1_split:complete_ends((yblock+1)*axis1_split,inp_arr.shape[1],num_blocks),:]=np.fft.fft(inp_arr[xblock*axis0_split:complete_ends((xblock+1)*axis0_split,inp_arr.shape[0],num_blocks),yblock*axis1_split:complete_ends((yblock+1)*axis1_split,inp_arr.shape[1],num_blocks),:],axis=2) 
	def gauss(self,x,mean,fw_hm):
			return np.exp(-(x-mean)**2/fw_hm**2*2.7725887222397811)
	def calc_templ(self,N1,N2,N3,templ_freq,spat_fwhm):
		"""

		This function calculates the FFT of a 3D Gaussian template.

		Parameters
		----------
		N1,N2,N3: int or double
			The sizes along three axes of the data cube.
		templ_freq, spat_fwhm: double
			The Gaussian template FWHM along the frequency and spatial dimension, in channels/pixels.

		Returns
		-------
		np.ndarray of double 
			The FFT of the 3D Gaussian template, with size N1,N2,N3

		"""
		fwhm=2.355
		if spat_fwhm==0:
			s1=1e-4/fwhm
			s2=1e-4/fwhm
		else:
			s1=spat_fwhm/fwhm
			s2=spat_fwhm/fwhm
		s3=templ_freq/fwhm
		r1=N1/(s1*2*np.pi)
		r2=N2/(s2*2*np.pi)
		r3=N3/(s3*2*np.pi)
		normaliz=np.sqrt(N1*N2*N3/np.sqrt(np.pi)**3/(r1*r2*r3))
		y,x=np.indices((N2,N1))
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
		r = np.hypot(x - center[0], y - center[1])
		spat=np.exp(-r**2/2/(r1)**2)
		my_version=np.reshape(np.tile(spat,(int(N3),1)),(int(N3),N1,N2))
		argh=np.array([self.gauss(x,N3/2,r3*fwhm) for x in range(int(N3))])
		my_version=np.einsum('i,ikl->ikl',argh,my_version)
		my_version*=normaliz
		return np.fft.ifftshift(my_version)
	def inverse_FFT(self,myarray,spat_width):
		"""

		This function calculates the inverse FFT of a large data cube, along all 3 axes. 

		Parameters
		----------
		myarray: np.ndarray of double
			The array to be FFT'ed, typically will be a memmap for handling large data volumes
		spat_width: int
			The spatial template size, to locate the appropriate working folder

		Returns
		-------

		"""
		ax0=np.memmap(self.working_directory + 'MF_'+str(spat_width)+'pix/ax0',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		self.my_fft(myarray,ax0,0)
		gc.collect()
		del myarray
		ax01=np.memmap(self.working_directory + 'MF_'+str(spat_width)+'pix/ax01',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		self.my_fft(ax0,ax01,1)
		del ax0
		gc.collect()
		os.system('rm ' + self.working_directory + 'MF_'+str(spat_width)+'pix/ax0')
		ax012=np.memmap(self.working_directory + 'MF_'+str(spat_width)+'pix/ax012',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		self.my_fft(ax01,ax012,2)
		del ax01
		os.system('rm ' + self.working_directory + 'MF_'+str(spat_width)+'pix/ax01')
		peaks=np.memmap(self.working_directory + 'MF_'+str(spat_width)+'pix/temp_peaks',dtype='float32',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		peaks[:]=np.real(ax012)/self.Nchan/self.Nypix/self.Nxpix
		del ax012
		os.system('rm ' + self.working_directory + 'MF_'+str(spat_width)+'pix/ax012')
		return peaks
	def match_filter(self,freq_width,spat_width):
		"""

		This function calculates the Convolution of a 3D Gaussian template with the S/N data cube.

		Parameters
		----------
		freq_width,spat_width: int 
			The Gaussian template FWHM along the frequency and spatial dimension, in channels/pixels.

		Returns
		-------

		"""
		N1=self.Nxpix
		N2=self.Nypix
		N3=self.Nchan
		FTt=np.memmap(self.working_directory + 'FTSNR_conjugated',dtype='complex64',mode='r',shape=(self.Nchan, self.Nypix,self.Nxpix))
		FTt2=np.memmap(self.working_directory + 'MF_'+str(spat_width)+'pix/templ'+str(int(freq_width)),dtype='float32',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))  #changed mode='w+' to 'r' for templates already existing
		FTt2[:]=self.calc_templ(N1,N2,N3,freq_width,spat_width)  #comment this if templ already there
		print('template done')
		prod=np.memmap(self.working_directory + 'MF_'+str(spat_width)+'pix/prod_temp',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		prod[:]=FTt*FTt2
		print('product done')
		del FTt
		del FTt2
		gc.collect()
		peaks=self.inverse_FFT(prod,spat_width)
		np.save(self.working_directory + 'MF_'+str(spat_width)+'pix/peaks'+str(int(freq_width))+'_full3d',peaks)
		del peaks
		os.system('rm ' + self.working_directory + 'MF_'+str(spat_width)+'pix/temp_peaks')
		os.system('rm ' + self.working_directory + 'MF_'+str(spat_width)+'pix/prod_temp')
		print('template '+str(int(freq_width))+' succesfully finished')
	def run_frequencies_parallel(self,spat_width):
		"""
		
		This function runs the Matched Filtering on the different frequency widths, for a fixed spatial size.
		Right now it is executing them serially, but can easily be modified to run them in parallel if needed.
		It is useful to run them as individual processes because the memory allocation is then freed upon completion.

		Parameters
		----------
		spat_width: int 
			The Gaussian template FWHM along the spatial dimension, in channels/pixels.

		Returns
		-------


		"""
		from multiprocessing import Process,Pool
		for width in self.freq_fwhms:
			p=Process(target=self.match_filter, args=(width,spat_width,))
			p.start()
			p.join()
	def run_allspat(self):
		"""

		This function runs the Matched Filtering on the every spatial size, creating the appropriate working folder.

		Parameters
		----------
		Returns
		-------


		"""
		for spat_fwh in self.spatial_fwhms:
			if not os.path.isdir(self.working_directory + 'MF_'+str(spat_fwh)+'pix'):
				os.mkdir(self.working_directory + 'MF_'+str(spat_fwh)+'pix')
			self.run_frequencies_parallel(spat_fwh)
	def make_combined_negatives(self):
		"""

		This function finds the peaks within each Matched Filtered data cube (i.e., for each template) above SNR threshold=4. Searches for negative lines.

		Parameters
		----------
		Returns
		-------
		'working/combined_dict_pickled_negative' contains the pickled python dictionary information with the positions of all the peaks, for each template combination (as keys).

		"""
		def find_tops_and_clump(peaks,tops,thresh,tag,conv_factor):
			hipoints_chan,hipoints_y,hipoints_x=np.where(peaks<thresh*conv_factor) 
			temp_tops=[]
			for obj_id,chan in enumerate(hipoints_chan):
				value=peaks[chan,hipoints_y[obj_id],hipoints_x[obj_id]]
				freq_rang=6				#you can change these, make sure to also change the corresponding variable inside make_combined_positive
				spat_rang=6   			#again, for the spatial radius
				startchan=np.max([0,chan-freq_rang])
				endchan=np.min([self.Nchan-1,chan+freq_rang])
				startx=np.max([0,hipoints_x[obj_id]-spat_rang])
				endx=np.min([self.Nxpix-1,hipoints_x[obj_id]+spat_rang])
				starty=np.max([0,hipoints_y[obj_id]-spat_rang])
				endy=np.min([self.Nypix-1,hipoints_y[obj_id]+spat_rang])
				if np.min(peaks[startchan:endchan,starty:endy,startx:endx])>=value: 
					temp_tops.append((value/conv_factor,(hipoints_x[obj_id],hipoints_y[obj_id],chan)))
			tops[tag]=temp_tops
		def load_and_run(spat_width,freq_width,combined):
			peak=np.load(self.working_directory + 'MF_'+str(spat_width)+'pix/peaks'+str(freq_width)+'_full3d.npy')
			peaks=peak.astype('float64')
			del peak
			conv_factor=np.std(peaks[SNR!=0])
			find_tops_and_clump(peaks,combined,-4,(spat_width,freq_width),conv_factor)
			del peaks
			return conv_factor		
		combined=dict()
		conv_factors=[]
		with open(self.working_directory + 'conv_factor.dat','w') as outp_conv_factor:
			SNR=self.SNR
			for spat in self.spatial_fwhms:
				for fre in self.freq_fwhms:
					conv_factor=load_and_run(spat,fre,combined)
					print(conv_factor, file=outp_conv_factor)
					conv_factors.append(conv_factor)
					print("done",spat," ",fre)
		with open(self.working_directory + 'combined_dict_pickled_negative','wb') as outpfil:
			pickle.dump(combined,outpfil,protocol=-1)
		return conv_factors
	def make_combined_positives(self,conv_factors):
		"""

		This function finds the peaks within each Matched Filtered data cube (i.e., for each template) above SNR threshold=4. Searches for positive lines.
		The difference to the negatives is that there is no need to measure the Standard deviation of the Matched Filtered cubes anymore, so those values are re-utilized.

		Parameters
		----------
		conv_factors: list of int
			The noise level (std) of the Matched Filtered cubes, measured by make_combined_negatives
		
		Returns
		-------
		'working/combined_dict_pickled_positive' contains the pickled python dictionary information with the positions of all the peaks, for each template combination (as keys).

		"""
		def find_tops_and_clump(peaks,tops,thresh,tag,conv_factor):
			hipoints_chan,hipoints_y,hipoints_x=np.where(peaks>thresh*conv_factor) 
			temp_tops=[]
			for obj_id,chan in enumerate(hipoints_chan):
				value=peaks[chan,hipoints_y[obj_id],hipoints_x[obj_id]]
				freq_rang=6				#you can change these, make sure to also change the corresponding variable inside make_combined_negative
				spat_rang=6				#again, for the spatial radius
				startchan=np.max([0,chan-freq_rang])
				endchan=np.min([self.Nchan-1,chan+freq_rang])
				startx=np.max([0,hipoints_x[obj_id]-spat_rang])
				endx=np.min([self.Nxpix-1,hipoints_x[obj_id]+spat_rang])
				starty=np.max([0,hipoints_y[obj_id]-spat_rang])
				endy=np.min([self.Nypix-1,hipoints_y[obj_id]+spat_rang])
				if np.max(peaks[startchan:endchan,starty:endy,startx:endx])<=value: #max to min and <= to >=
					temp_tops.append((value/conv_factor,(hipoints_x[obj_id],hipoints_y[obj_id],chan)))
			tops[tag]=temp_tops
		def load_and_run(spat_width,freq_width,combined,conv_factor):
			peak=np.load(self.working_directory + 'MF_'+str(spat_width)+'pix/peaks'+str(freq_width)+'_full3d.npy')
			peaks=peak.astype('float64')
			del peak
			find_tops_and_clump(peaks,combined,4,(spat_width,freq_width),conv_factor)
			del peaks
		combined=dict()
		SNR=self.SNR
		idx=0
		for spat in self.spatial_fwhms:
			for fre in self.freq_fwhms:
				conv_factor=conv_factors[idx]
				load_and_run(spat,fre,combined,conv_factor)
				print("done",spat," ",fre)
				idx+=1
		with open(self.working_directory + 'combined_dict_pickled_positive','wb') as outpfil:
			pickle.dump(combined,outpfil,protocol=-1)
	def make_reduint_positive(self):
		"""

		This function combines the peaks from different templates, identifying them as corresponding to the same feature if falling within a local cluster. It calculates the moving average position of the peaks cluster and adopts the highest SNR as the feature SNR. The template which maximizes this SNR is the Matched filter and its size is saved in the output data structure.

		Parameters
		----------
		Returns
		-------
		'working/reduint_positive.dat' contains the pickled python dictionary information with the positions of all the merged peaks, the peak SNR and the template size which maximizes SNR.

		"""
		def distsq(xxx_todo_changeme, xxx_todo_changeme2):
			#return (1.*a-1.*d)**2+(1.*b-1.*e)**2+(1.*c-1.*f)**2
			(a,b,c) = xxx_todo_changeme
			(d,e,f) = xxx_todo_changeme2
			return (1.*a-1.*d)**2+(1.*b-1.*e)**2<25. and np.absolute(1.*c-1.*f)<15.     #You can change these, both the spatial radius^2 and channel thresholds which control line features merging across different templates
		with open(self.working_directory + 'combined_dict_pickled_positive','rb') as inp:
			combined=pickle.load(inp)
		tag_combined=[(x[0],x[1],key) for key in list(combined.keys()) for x in combined[key]]
		tag_combined.sort(key=operator.itemgetter(0),reverse=True)
		redu=[]
		reduced=[]
		for obj in tag_combined:
				found=False
				for idx,red in enumerate(reduced):
					if distsq(redu[idx][1],obj[1]) and redu[idx][2]!=obj[2]:  
						red.append(obj)
						if redu[idx][0]<obj[0]:
							redu[idx]=(obj[0],tuple(np.average([x[1] for x in red],axis=0,weights=[x[0]**2 for x in red])),obj[2])
						else:
							redu[idx]=(redu[idx][0],tuple(np.average([x[1] for x in red],axis=0,weights=[x[0]**2 for x in red])),redu[idx][2])
						found=True
				if not found:
						reduced.append([obj])
						redu.append(obj)
		reduint=[(x[0],(int(round(x[1][0])),int(round(x[1][1])),int(round(x[1][2]))   ),len(reduced[idx]),x[2]) for idx,x in enumerate(redu)]
		reduced_sorted=[x for (y,x) in sorted(zip(reduint,reduced), key=lambda k_v: operator.itemgetter(0)(k_v[1]),reverse=True)]
		reduint.sort(key=operator.itemgetter(0),reverse=True)
		with open(self.working_directory + 'reduint_positive.dat','wb') as outp:
			pickle.dump(reduint,outp,protocol=-1)
		return reduint
	def make_reduint_negative(self):
		"""

		This function combines the peaks from different templates, identifying them as corresponding to the same feature if falling within a local cluster. It calculates the moving average position of the peaks cluster and adopts the highest SNR as the feature SNR. The template which maximizes this SNR is the Matched filter and its size is saved in the output data structure.

		Parameters
		----------
		Returns
		-------
		'working/reduint_negative.dat' contains the pickled python dictionary information with the positions of all the merged peaks, the peak SNR and the template size which maximizes SNR.

		"""
		def distsq(xxx_todo_changeme3, xxx_todo_changeme4):
			#return (1.*a-1.*d)**2+(1.*b-1.*e)**2+(1.*c-1.*f)**2
			(a,b,c) = xxx_todo_changeme3
			(d,e,f) = xxx_todo_changeme4
			return (1.*a-1.*d)**2+(1.*b-1.*e)**2<25. and np.absolute(1.*c-1.*f)<15.   #You can change these, both the spatial radius^2 and channel thresholds which control line features merging across different templates
		with open(self.working_directory + 'combined_dict_pickled_negative','rb') as inp:
			combined=pickle.load(inp)
		tag_combined=[(x[0],x[1],key) for key in list(combined.keys()) for x in combined[key]]
		tag_combined.sort(key=operator.itemgetter(0),reverse=False)
		redu=[]
		reduced=[]
		for obj in tag_combined:
				found=False
				for idx,red in enumerate(reduced):
					if distsq(redu[idx][1],obj[1]) and redu[idx][2]!=obj[2]:  #was34; was 28
						red.append(obj)
						if redu[idx][0]>obj[0]:
							redu[idx]=(obj[0],tuple(np.average([x[1] for x in red],axis=0,weights=[x[0]**2 for x in red])),obj[2])
						else:
							redu[idx]=(redu[idx][0],tuple(np.average([x[1] for x in red],axis=0,weights=[x[0]**2 for x in red])),redu[idx][2])
						found=True
				if not found:
						reduced.append([obj])
						redu.append(obj)
		reduint=[(x[0],(int(round(x[1][0])),int(round(x[1][1])),int(round(x[1][2]))   ),len(reduced[idx]),x[2]) for idx,x in enumerate(redu)]
		reduced_sorted=[x for (y,x) in sorted(zip(reduint,reduced), key=lambda k_v1: operator.itemgetter(0)(k_v1[1]),reverse=False)]
		reduint.sort(key=operator.itemgetter(0),reverse=False)
		with open(self.working_directory + 'reduint_negative.dat','wb') as outp:
			pickle.dump(reduint,outp,protocol=-1)
		return reduint
	def remove_cont_sources(self,cont_source,reduint):
		"""

		This function allows removing line candidates which are too close to a continuum source and are therefore likely to be noise on top of continuum emission

		Parameters
		----------
		cont_source: list of triplets of int
			list of continuum source coordinates in (xpix,ypix,chan) form
		reduint: dictionary of lists
			list of all line candidates in standard format defined above

		Returns
		-------
		reduint_no_cont: dictionary of lists
			list of all line candidates in standard format, with sources removed if lying spatially too close to continuum source

		"""
		reduint_no_cont=[]
		for obj in reduint:
			cont=False
			for co_id in cont_source:
				if (obj[1][0]-co_id[0])**2+(obj[1][1]-co_id[1])**2<25:
					cont=True
			if not cont:
				reduint_no_cont.append(obj)
		return reduint_no_cont
	def run(self):
		"""

		This function runs a typical Matched Filtering routine from start to end.

		Parameters
		----------
		Returns
		-------

		"""
		self.make_FTSNR()
		self.run_allspat()
		conv_factors=self.make_combined_negatives()
		self.make_combined_positives(conv_factors)
		self.reduint_pos=self.make_reduint_positive()
		self.reduint_neg=self.make_reduint_negative()





