# !/usr/bin/env python
# encoding:utf-8

# Lists all the saved classifier models
# Insikt Intelligence S.L. 2019
 

import os


def claslisting(user_id,case_id):
	
	path = './data/probability/insikt/'

	files = []
	
			
	if len(user_id) ==0 and len(case_id)==0:
		print('1')
		for r, d, f in os.walk(path):
			print(f)
			for file in f:
				#if 'Insikt' in file:
				#	print('hello')
				files.append(file)
			
	if len(user_id) ==0 and len(case_id)> 0:
		print('2')
		for r, d, f in os.walk(path):
			for file in f:
				if case_id in file:
					files.append(file)
				if 'Insikt' in file:
					files.append(file)

	if len(user_id) >0 and len(case_id)==0:
		print('3')
		for r, d, f in os.walk(path):
			for file in f:
				if user_id in file:
					files.append(file)
				if 'Insikt' in file:
					files.append(file)

	if len(user_id) >0 and len(case_id)>0:
		print('4')
		for r, d, f in os.walk(path):
			
			for file in f:
				
				if user_id and case_id in file:
					files.append(file)
				if 'Insikt' in file:
					
					files.append(file)

	return(files)
	#for f in list(files):
	#	return(f)

	

	


