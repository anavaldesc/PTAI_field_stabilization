#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:56:33 2020

@author: banano
"""


import numpy as np
import pandas as pd
import os
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import nrefocus

def matchfiles(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if fnmatch(file, '*.h5'):
                yield root, file
        break  # break after listing top level dir


def getfolder(date, sequence):
    date = pd.to_datetime(str(date))
    folder = 'data/' + date.strftime('%Y/%m/%d') + '/{:04d}'.format(sequence)
    return folder

def get_uwavelock_images(camera,
                         date,
                         sequence,
                         sequence_type,
                         image_keys,
                         scanned_parameter='ARP_FineBias',
                         redo_prepare=True):
    
    folder = getfolder(date, sequence)
    print('Your current working directory is %s' %os.getcwd())
    
    outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)
    
    try:
        with h5py.File('processed_data/' + outfile, 'r') as f:
            f['data']
            
    except KeyError:
        redo_prepare = True
        print('Data not found, preparing again..')
    except IOError:
        redo_prepare = True
        print('Data not found, preparing again..')

    
    if redo_prepare:
        
        print('Looking for files in %s' %folder)
        scanned_parameters = []
        img_list = [] 
        run_number = []
        n_runs = []
        sequence_id = []
        sequence_index = []
    

        try:
            i = 0
            j=0
            for r, file in matchfiles(folder):
                i += 1;
            print('Found %s files in folder'%i)
              
            if i> 0:
                
                print('Preparing {} data...'.format(sequence_type)) 
                for r, file in tqdm(matchfiles(folder)):
                    j+=1
                    with h5py.File(os.path.join(r, file), 'r') as h5_file:
                 
                        img = np.empty((len(image_keys), 484, 644))
                        
                        try:
                            for i, key in enumerate(image_keys):
                                img[i] = h5_file['data']['images' + camera][key][:]
                        
                        except Exception as e: 
                            print(e)
                            img*= np.nan
                            
                        try:
                            attrs = h5_file['globals'].attrs
                            scanned_parameters.append(attrs[scanned_parameter])  
                            attrs = h5_file.attrs
                            run_number.append(attrs['run number'])
#                     
#                            sequence_id.append(attrs['sequence_id'])
                            sequence_index.append(attrs['sequence_index'])
                        
                        except:
                            print(j)
                            scanned_parameters.append(np.nan)  
                            run_number.append(np.nan)
                            n_runs.append(np.nan)
                            sequence_id.append(np.nan)
                            sequence_index.append(np.nan)
                        
                        img = np.float64(img)
                        img_list.append(img)
                        
            df = pd.DataFrame()
            df[scanned_parameter] = scanned_parameters
            df['run_number'] = run_number
            df = df.sort_values(by=scanned_parameter)
            img_list = np.array(img_list)
            sorting_indices = df.index.values
            sorted_img = np.empty_like(img_list)

            for i,idx in enumerate(sorting_indices):
                sorted_img[i] = img_list[idx]

            print('Saving data...')
    
            hf = h5py.File('processed_data/' + outfile, 'w')
            hf.attrs['sequence_id'] = sequence_id[0]
            hf.attrs['n_runs'] = n_runs[0]
            hf.attrs['sequence_index'] = sequence_index[0]
            
            g1 = hf.create_group('sorted_images')
            g1.create_dataset('sorted_img',data=sorted_img, compression="gzip")
#            g1.create_dataset('sorted_atoms', data=sorted_atoms, compression='gzip')
            g2 = hf.create_group('data')
            
            
            for key in tqdm(df.keys()):
                try:
                    g2.create_dataset(str(key), data=df[key].values)
                except Exception as e:
#                    print(e)
                    g2.create_dataset(str(key), data=df[key].values.astype('S'))
                    
            hf.close()
#            df.to_hdf('results/' + outfile, 'data', mode='w')
            
        except Exception as e:
#            print('Fix your analysis')
            print(e)        

#    
    else:
        
        print('Loading processed data...')
        df = pd.DataFrame()
        hf = h5py.File('processed_data/' + outfile, mode='r')
        try: 
            g1 = hf.get('sorted_images')
            sorted_img = g1.get('sorted_img')
            sorted_img = np.array(sorted_img)
            g2 = hf.get('data')
            for key in g2.keys():
                df[key] = np.array(g2.get(key))
            hf.close()
        except:
            print('Could not find data')
            hf.close()
        
    return df, sorted_img

def data_processing(camera, 
                    date, 
                    sequence, 
                    sequence_type, 
                    scanned_parameter='ARP_FineBias',
                    redo_prepare=True):
    
    """
    Takes a camera, data and sequence number and returns a dataframe with 
    information such as run number, integrated od, scanned variable, microwave
    lock paremeters and an array with 
    """
        
    folder = getfolder(date, sequence)
    print('Your current working directory is %s' %os.getcwd())
    
    outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)
    
    try:
        with h5py.File('processed_data/' + outfile, 'r') as f:
            f['data']
            
    except KeyError:
        redo_prepare = True
        print('Data not found, preparing again..')
    except IOError:
        redo_prepare = True
        print('Data not found, preparing again..')

    
    if redo_prepare:
        
        print('Looking for files in %s' %folder)
        scanned_parameters = []
        img_list = [] 
        run_number = []
        n_runs = []
        sequence_id = []
        sequence_index = []
        status = []
        err = []
        od1 = []
        od2 = []

        try:
            i = 0
            j=0
            for r, file in matchfiles(folder):
                i += 1;
            print('Found %s files in folder'%i)
              
            if i> 0:
                
                print('Preparing {} data...'.format(sequence_type)) 
                for r, file in tqdm(matchfiles(folder)):
                    j+=1
                    with h5py.File(os.path.join(r, file), 'r') as h5_file:
                 
                        try:
                            uwaves = h5_file['globals/uwaves'].attrs
                            uwaves_lock = uwaves.get('UWAVES_LOCK', 'False')
                            uwaves_lock = eval(uwaves_lock) #convert string to boolean
                            
                        except KeyError:
                            uwaves_lock = False
                            
                        try:
                            img = h5_file['data']['images' + camera]['Raw'][:]
                        
                        except: 
                            img = np.ones([9,484, 644]) * np.nan
                            
                        try:
                            attrs = h5_file['globals'].attrs
                            scanned_parameters.append(attrs[scanned_parameter])  
                            attrs = h5_file.attrs
                            run_number.append(attrs['run number'])
                            n_runs.append(attrs['n_runs'])
                            sequence_id.append(attrs['sequence_id'])
                            sequence_index.append(attrs['sequence_index'])
                        
                        except:
                            print(j)
                            scanned_parameters.append(np.nan)  
                            run_number.append(np.nan)
                            n_runs.append(np.nan)
                            sequence_id.append(np.nan)
                            sequence_index.append(np.nan)
                        
                        img = np.float64(img)
                        img_list.append(img)
                        
                        try:

                            uwave_attrs = h5_file['results/uwave_lock'].attrs
                            od1.append(uwave_attrs['od1'])
                            od2.append(uwave_attrs['od2'])
                            err.append(uwave_attrs['err'])                        
                            
                        except Exception as e:
#                            status.append(str('bad shot'))
#                            print(e)
#                            ods.append(np.nan * np.ones([644, 484]))
#                            probe_list.append(np.nan * np.ones([644, 484]))
#                            atoms_list.append(np.nan * np.ones([644, 484]))
#                            integrated_od.append(np.nan)
                            od1.append(np.nan)
                            od2.append(np.nan)
                            err.append(np.nan)

            df = pd.DataFrame()
            df[scanned_parameter] = scanned_parameters
            df['run_number'] = run_number
#            df['n_runs'] = n_runs
#            df['sequence_id'] = sequence_id
#            df['sequence_index'] = sequence_index
#            df['status'] = status
#            df['status'] = df['status'].astype('str') 
#            df['integrated_od'] = integrated_od
            df['od1'] = od1
            df['od2'] = od2
            df['err'] = err
            
#            print('Here') 

    #        df = df.dropna()
            df = df.sort_values(by=scanned_parameter)
#            print(df[scanned_paramteer].values)
            img_list = np.array(img_list)
#            probe_list = np.array(probe_list)
#            atoms_list = np.array(atoms_list)
            sorting_indices = df.index.values
#            print(ods.shape)
            sorted_img = np.empty_like(img_list)
#            print(len(sorted_img))
#            print(len(sorting_indices))

           

            for i,idx in enumerate(sorting_indices):
                sorted_img[i] = img_list[idx]

            
            print('Saving data...')
    
            hf = h5py.File('processed_data/' + outfile, 'w')
            hf.attrs['sequence_id'] = sequence_id[0]
            hf.attrs['n_runs'] = n_runs[0]
            hf.attrs['sequence_index'] = sequence_index[0]
            
            g1 = hf.create_group('sorted_images')
            g1.create_dataset('sorted_img',data=sorted_img, compression="gzip")
#            g1.create_dataset('sorted_atoms', data=sorted_atoms, compression='gzip')
            g2 = hf.create_group('data')
            
            
            for key in tqdm(df.keys()):
                try:
                    g2.create_dataset(str(key), data=df[key].values)
                except Exception as e:
#                    print(e)
                    g2.create_dataset(str(key), data=df[key].values.astype('S'))
                    
            hf.close()
#            df.to_hdf('results/' + outfile, 'data', mode='w')
            
        except Exception as e:
#            print('Fix your analysis')
            print(e)        

#    
    else:
        
        print('Loading processed data...')
        df = pd.DataFrame()
        hf = h5py.File('processed_data/' + outfile, mode='r')
        try: 
            g1 = hf.get('sorted_images')
            sorted_img = g1.get('sorted_img')
            sorted_img = np.array(sorted_img)
    #        sorted_atoms = g1.get('sorted_atoms')
    #        sorted_atoms = np.array(sorted_atoms)
            g2 = hf.get('data')
            for key in g2.keys():
                df[key] = np.array(g2.get(key))
            hf.close()
        except:
            print('Could not find data')
            hf.close()
        
    return df, sorted_img#, sorted_atoms#, sorted_probe


def simple_data_processing(date, sequence, 
                           scanned_parameter='Raman_pulse_time'):
    

    """
    Takes a camera, data and sequence number and returns a dataframe with 
    information such as run number, integrated od, scanned variable, microwave
    lock paremeters and an array with 
    """

    folder = getfolder(date, sequence)
    print('Your current working directory is %s' %os.getcwd())
    
        
    print('Looking for files in %s' %folder)
    scanned_parameters = []
    int_od = []
    od1 = []
    od2 = []
    err = []
     
    try:
        i = 0
        j=0
        for r, file in matchfiles(folder):
            i += 1;
        print('Found %s files in folder'%i)
          
        if i> 0:
            
            print('Preparing data...') 
            for r, file in tqdm(matchfiles(folder)):
                j+=1
                with h5py.File(os.path.join(r, file), 'r') as h5_file:
                                        
                    try:
#                        print(j)

                        rois_od_attrs = h5_file['results/uwave_lock'].attrs
                        od1.append(rois_od_attrs['od1'])
                        od2.append(rois_od_attrs['od2'])
                        err.append(rois_od_attrs['err'])

                        attrs = h5_file['globals'].attrs
                        if scanned_parameter == 'delta_xyz_freqs':
                            scanned_parameters.append(attrs[scanned_parameter][0])
                        else:
                            scanned_parameters.append(attrs[scanned_parameter])
                    
                    except:

                        scanned_parameters.append(np.nan)  
                        od1.append(np.nan)
                        od2.append(np.nan)
                        err.append(np.nan)
#                        print(scanned_parameters)
                        int_od.append(np.nan)
                    
        df = pd.DataFrame()
        df[scanned_parameter] = scanned_parameters
#        print(df)
        df['od1'] = od1
        df['od2'] = od2
        df['err'] = err
        df = df.sort_values(by=scanned_parameter)
    
    except Exception as e:
        print(e)
#        print(len(roi_0))
#        print(len(scanned_parameters))
           
    return df


def rabi_peak(pars,delta,data=None):
    
    params = pars.valuesdict()
    delta_shift = delta - params['delta0']
    y = np.sin(np.sqrt(params['om']**2 + delta_shift**2)/2 * params['t'])**2
    y *= params['om']**2 / (params['om']**2 + delta_shift**2) 
    y *= params['amp']
    y += params['off']
    
    if data is None:
        return y
    
    else:
        return y - data
    
def gaussian2D(xy_vals, bkg, amp, x0, sigmax, y0, sigmay) :

    gauss2D = bkg + amp*np.exp(-1*(xy_vals[:,1]-x0)**2/(2*sigmax**2)
                                -1*(xy_vals[:, 0]-y0)**2/(2*sigmay**2))
    
    return gauss2D

def make_xy_grid(image):
    
    x, y = image.shape
    data = np.empty((x * y, 2))
    x = np.arange(x)
    y = np.arange(y)
    xx, yy = np.meshgrid(x, y)
    data[:,0] = xx.flatten()
    data[:,1] = yy.flatten()
    
    return data

def refocus(img, pixels):
    img_sqrt = np.sqrt(img.clip(0))
    img_temp = nrefocus.refocus(img_sqrt, pixels, 1, 1)
    return (img_temp * np.conjugate(img_temp)).real

def puck(xyVals, x0, wx, y0, wy) :
    # returns 1.0 within a specified boundary. 0.0 everywhere else:
    #   X Central value                : x0
    #   X width / 2                       : wx
    #   Y Central value                : y0
    #   Y width / 2                        : wy
    condition = (1.0 - ((xyVals[0]-x0)/wx)**2.0 - ((xyVals[1]-y0)/wy)**2.0);
    condition[condition < 0.0] = 0.0;
    condition[condition > 0.0] = 1.0;
    return condition;