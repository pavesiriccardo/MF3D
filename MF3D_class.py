#should take SNR map and Match filter, and return list
import pyfits,gc
import numpy as np
import operator,cPickle
import os

class MF3D(object):
	def dist3d(self,pos1,pos2):
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)
	def dist2d(self,pos1,pos2):
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
	def __init__(self,SNR,spatial_fwhms,freq_fwhms):   #SNR is numpy array, no degenerate axes please
		self.SNR=SNR
		if len(self.SNR.shape)>3:
			print 'More than 3 dimensions! Need to drop degenerate axes before giving me SNR cube'
		self.Nchan,self.Nypix,self.Nxpix=self.SNR.shape
		if self.Nypix!=self.Nxpix:
			print 'So far can only use square images, please fix your cube to spatially square'
		self.SNR=np.where(np.isnan(self.SNR),0,self.SNR)
		self.SNR=np.where(self.SNR==1.,0,self.SNR)
		self.spatial_fwhms=spatial_fwhms
		self.freq_fwhms=freq_fwhms
	def make_FTSNR(self):   #this makes the complex conjugate of the FT of the SNR, need that because the convolution requires IFT(product) which is equivalent to FT(prod_conjugate)_conjugate/Npix, the template and the final are real so those are not affected by conjugate
		os.mkdir('working')
		ax0=np.memmap('working/ax0',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		#ax0[:]=np.fft.fftn(self.SNR,axes=[0])
		self.my_fft(self.SNR,ax0,0)
		ax01=np.memmap('working/ax01',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		#ax01[:]=np.fft.fftn(ax0,axes=[1])
		self.my_fft(ax0,ax01,1)
		del ax0
		os.system('rm working/ax0')
		ax012=np.memmap('working/ax012',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		#ax012[:]=np.fft.fftn(ax01,axes=[2])
		self.my_fft(ax01,ax012,2)
		del ax01
		os.system('rm working/ax01')
		FTt=np.memmap('working/FTSNR_conjugated',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		FTt[:]=np.conjugate(ax012)
		os.system('rm working/ax012')
		del ax012
	def my_fft(self,inp_arr,outp_arr,axis):
		num_blocks=2        #How many blocks (per axis) do we want to split the cube into, to reduce memory usage. Keep it as low as your RAM allows you to (2 or 3, perhaps)
		def complete_ends(index,tot_length,error):
			if tot_length-index<error:
				return tot_length
			else:
				return index
		axis0_split=inp_arr.shape[0]/num_blocks
		axis1_split=inp_arr.shape[1]/num_blocks
		axis2_split=inp_arr.shape[2]/num_blocks
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
		ax0=np.memmap('working/MF_'+str(spat_width)+'pix/ax0',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		self.my_fft(myarray,ax0,0)
		gc.collect()
		del myarray
		ax01=np.memmap('working/MF_'+str(spat_width)+'pix/ax01',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		self.my_fft(ax0,ax01,1)
		del ax0
		gc.collect()
		os.system('rm working/MF_'+str(spat_width)+'pix/ax0')
		ax012=np.memmap('working/MF_'+str(spat_width)+'pix/ax012',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		self.my_fft(ax01,ax012,2)
		del ax01
		os.system('rm working/MF_'+str(spat_width)+'pix/ax01')
		peaks=np.memmap('working/MF_'+str(spat_width)+'pix/temp_peaks',dtype='float32',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		peaks[:]=np.real(ax012)/self.Nchan/self.Nypix/self.Nxpix
		del ax012
		os.system('rm working/MF_'+str(spat_width)+'pix/ax012')
		return peaks
	def match_filter(self,freq_width,spat_width):
		N1=self.Nxpix
		N2=self.Nypix
		N3=self.Nchan
		FTt=np.memmap('working/FTSNR_conjugated',dtype='complex64',mode='r',shape=(self.Nchan, self.Nypix,self.Nxpix))
		FTt2=np.memmap('working/MF_'+str(spat_width)+'pix/templ'+str(int(freq_width)),dtype='float32',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))  #changed mode='w+' to 'r' for templates already existing
		FTt2[:]=self.calc_templ(N1,N2,N3,freq_width,spat_width)  #comment this if templ already there
		print 'template done'
		prod=np.memmap('working/MF_'+str(spat_width)+'pix/prod_temp',dtype='complex64',mode='w+',shape=(self.Nchan, self.Nypix,self.Nxpix))
		prod[:]=FTt*FTt2
		print 'product done'
		del FTt
		del FTt2
		gc.collect()
		peaks=self.inverse_FFT(prod,spat_width)
		np.save('working/MF_'+str(spat_width)+'pix/peaks'+str(int(freq_width))+'_full3d',peaks)
		del peaks
		os.system('rm working/MF_'+str(spat_width)+'pix/temp_peaks')
		os.system('rm working/MF_'+str(spat_width)+'pix/prod_temp')
		print 'template '+str(int(freq_width))+' succesfully finished'
	def run_frequencies_parallel(self,spat_width):
		from multiprocessing import Process,Pool
		for width in self.freq_fwhms:
			p=Process(target=self.match_filter, args=(width,spat_width,))
			p.start()
			p.join()
	def run_allspat(self):
		for spat_fwh in self.spatial_fwhms:
			os.mkdir('working/MF_'+str(spat_fwh)+'pix')
			self.run_frequencies_parallel(spat_fwh)
	def make_combined_negatives(self):
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
			peak=np.load('working/MF_'+str(spat_width)+'pix/peaks'+str(freq_width)+'_full3d.npy')
			peaks=peak.astype('float64')
			del peak
			conv_factor=np.std(peaks[SNR!=0])
			find_tops_and_clump(peaks,combined,-4,(spat_width,freq_width),conv_factor)
			del peaks
			return conv_factor		
		combined=dict()
		conv_factors=[]
		outp_conv_factor=open('working/conv_factor.dat','w')
		SNR=self.SNR
		for spat in self.spatial_fwhms:
			for fre in self.freq_fwhms:
				conv_factor=load_and_run(spat,fre,combined)
				print >>outp_conv_factor,conv_factor
				conv_factors.append(conv_factor)
				print "done",spat," ",fre
		outp_conv_factor.close()
		outpfil=open('working/combined_dict_pickled_negative','w')
		cPickle.dump(combined,outpfil)
		outpfil.close()
		return conv_factors
	def make_combined_positives(self,conv_factors):
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
			peak=np.load('working/MF_'+str(spat_width)+'pix/peaks'+str(freq_width)+'_full3d.npy')
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
				print "done",spat," ",fre
				idx+=1
		outpfil=open('working/combined_dict_pickled_positive','w')
		cPickle.dump(combined,outpfil)
		outpfil.close()
	def make_reduint_positive(self):
		def distsq((a,b,c),(d,e,f)):
			#return (1.*a-1.*d)**2+(1.*b-1.*e)**2+(1.*c-1.*f)**2
			return (1.*a-1.*d)**2+(1.*b-1.*e)**2<25. and np.absolute(1.*c-1.*f)<15.     #You can change these, both the spatial radius^2 and channel thresholds which control line features merging across different templates
		inp=open('working/combined_dict_pickled_positive','r')
		combined=cPickle.load(inp)
		tag_combined=[(x[0],x[1],key) for key in combined.keys() for x in combined[key]]
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
		reduced_sorted=[x for (y,x) in sorted(zip(reduint,reduced), key=lambda (k,v): operator.itemgetter(0)(v),reverse=True)]
		reduint.sort(key=operator.itemgetter(0),reverse=True)
		outp=open('working/reduint_positive.dat','w')
		cPickle.dump(reduint,outp)
		outp.close()
		return reduint
	def make_reduint_negative(self):
		def distsq((a,b,c),(d,e,f)):
			#return (1.*a-1.*d)**2+(1.*b-1.*e)**2+(1.*c-1.*f)**2
			return (1.*a-1.*d)**2+(1.*b-1.*e)**2<25. and np.absolute(1.*c-1.*f)<15.   #You can change these, both the spatial radius^2 and channel thresholds which control line features merging across different templates
		inp=open('working/combined_dict_pickled_negative','r')
		combined=cPickle.load(inp)
		tag_combined=[(x[0],x[1],key) for key in combined.keys() for x in combined[key]]
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
		reduced_sorted=[x for (y,x) in sorted(zip(reduint,reduced), key=lambda (k,v): operator.itemgetter(0)(v),reverse=False)]
		reduint.sort(key=operator.itemgetter(0),reverse=False)
		outp=open('working/reduint_negative.dat','w')
		cPickle.dump(reduint,outp)
		outp.close()
		return reduint
	def remove_cont_sources(self,cont_source,reduint):	
		reduint_no_cont=[]
		for obj in reduint:
			cont=False
			for co_id in cont_source:
				if (obj[1][0]-co_id[0])**2+(obj[1][1]-co_id[1])**2<25:
					cont=True
			if not cont:
				reduint_no_cont.append(obj)
		return reduint_no_cont
	def DOITALL(self):
		self.make_FTSNR()
		self.run_allspat()
		conv_factors=self.make_combined_negatives()
		self.make_combined_positives(conv_factors)
		self.reduint_pos=self.make_reduint_positive()
		self.reduint_neg=self.make_reduint_negative()





