#!/usr/bin/python3

from sklearn.decomposition import NMF
import pandas as pd
import numpy as np


def main():
	''' main function'''

	data, titles = get_nmf_features()
	NMF_documents(data, titles)


def NMF_documents(nmf_features, titles):
	''' Explores NMF features'''

	# The code in this quoted-out section is replaced by the 'get_nmf_features' function below
	'''
	# Create an NMF instance: model
	model = NMF(n_components=6)

	# Fit the model to articles
	model.fit(articles)

	# Transform the articles: nmf_features
	nmf_features = model.transform(articles)

	# Print the NMF features
	print(nmf_features)
	'''

###############################################

	# Create a pandas DataFrame: df
	df = pd.DataFrame(nmf_features, index=titles)

	# Print the row for 'Anne Hathaway'
	print(df.loc['Anne Hathaway'])

	# Print the row for 'Denzel Washington'
	print(df.loc['Denzel Washington'])

###############################################

	# Create a DataFrame: components_df
	components_df = pd.DataFrame(model.components_, columns=words)

	# Print the shape of the DataFrame
	print(components_df.shape)

	# Select row 3: component
	component = components_df.iloc[3]

	# Print result of nlargest
	print(component.nlargest())


def get_nmf_features():
	''' Creates array of nmf features'''

	data = np.array([
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.40465322e-01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.66604756e-01],
     [3.82058457e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.98646427e-01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.81739753e-01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.85517119e-01],
     [1.29290459e-02, 1.37889184e-02, 7.76313733e-03, 3.34487322e-02, 0.00000000e+00, 3.34521818e-01],
     [0.00000000e+00, 0.00000000e+00, 2.06738655e-02, 0.00000000e+00, 6.04486283e-03, 3.59061038e-01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.90976703e-01],
     [1.54274156e-02, 1.42817184e-02, 3.76628748e-03, 2.37111817e-02, 2.62619893e-02, 4.80774318e-01],
     [1.11738298e-02, 3.13676816e-02, 3.09480486e-02, 6.57000345e-02, 1.96677390e-02, 3.38288648e-01],
     [0.00000000e+00, 0.00000000e+00, 5.30710112e-01, 0.00000000e+00, 2.83679347e-02, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 3.56503042e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [1.20127255e-02, 6.50033756e-03, 3.12239769e-01, 6.09770818e-02, 1.13861318e-02, 1.92602102e-02],
     [3.93485615e-03, 6.24432234e-03, 3.42367241e-01, 1.10768934e-02, 0.00000000e+00, 0.00000000e+00],
     [4.63821398e-03, 0.00000000e+00, 4.34907414e-01, 0.00000000e+00, 3.84274924e-02, 3.08133140e-03],
     [0.00000000e+00, 0.00000000e+00, 4.83280609e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [5.65016594e-03, 1.83532464e-02, 3.76526384e-01, 3.25460935e-02, 0.00000000e+00, 1.13334438e-02],
     [0.00000000e+00, 0.00000000e+00, 4.80905320e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 9.01849274e-03, 5.50998244e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 4.65961434e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 1.14078934e-02, 2.08651998e-02, 5.17767327e-01, 5.81408435e-02, 1.37853805e-02],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.10475273e-01, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 5.60094760e-03, 0.00000000e+00, 4.22379854e-01, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.36751253e-01, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.98092036e-01, 0.00000000e+00, 0.00000000e+00],
     [9.88394018e-02, 8.60029454e-02, 3.91028856e-03, 3.81017532e-01, 4.39239942e-04, 5.22151481e-03],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.72169888e-01, 0.00000000e+00, 7.13541985e-03],
     [1.31468715e-02, 1.04851581e-02, 0.00000000e+00, 4.68906049e-01, 0.00000000e+00, 1.16309971e-02],
     [3.84548688e-03, 0.00000000e+00, 0.00000000e+00, 5.75710557e-01, 0.00000000e+00, 0.00000000e+00],
     [2.25244164e-03, 1.38734017e-03, 0.00000000e+00, 5.27945778e-01, 1.20264671e-02, 1.49483895e-02],
     [0.00000000e+00, 4.07541038e-01, 1.85711419e-03, 0.00000000e+00, 2.96610497e-03, 4.52342305e-04],
     [1.53421227e-03, 6.08162380e-01, 5.22269328e-04, 6.24853541e-03, 1.18444733e-03, 4.40074955e-04],
     [5.38819366e-03, 2.65012410e-01, 5.38501825e-04, 1.86925929e-02, 6.38651189e-03, 2.90104872e-03],
     [0.00000000e+00, 6.44904602e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 6.08896305e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 3.43679231e-01, 0.00000000e+00, 0.00000000e+00, 3.97794829e-03, 0.00000000e+00],
     [6.10508849e-03, 3.15307295e-01, 1.54877295e-02, 0.00000000e+00, 5.06244563e-03, 4.74335450e-03],
     [6.47373328e-03, 2.13324818e-01, 9.49479174e-03, 4.56981207e-02, 1.71914481e-02, 9.52063620e-03],
     [7.99147234e-03, 4.67586974e-01, 0.00000000e+00, 2.43425413e-02, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 6.42808850e-01, 0.00000000e+00, 2.35855413e-03, 0.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.77080113e-01, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.94253212e-01, 0.00000000e+00],
     [0.00000000e+00, 2.99063787e-04, 2.14484813e-03, 0.00000000e+00, 3.81776318e-01, 5.83780479e-03],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.64692771e-03, 5.42238170e-01, 0.00000000e+00],
     [1.78059654e-03, 7.84399946e-04, 1.41625538e-02, 4.59813501e-04, 4.24299848e-01, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.11388821e-01, 0.00000000e+00],
     [0.00000000e+00, 0.00000000e+00, 3.28381076e-03, 0.00000000e+00, 3.72884543e-01, 0.00000000e+00],
     [0.00000000e+00, 2.62079735e-04, 3.61098295e-02, 2.32336069e-04, 2.30509332e-01, 0.00000000e+00],
     [1.12517699e-02, 2.12323850e-03, 1.60969841e-02, 1.02484866e-02, 3.25459591e-01, 3.75880596e-02],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.18955707e-01, 3.57698689e-04],
     [3.08373460e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [3.68181576e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [3.97953206e-01, 2.81698167e-02, 3.67005937e-03, 1.70066802e-02, 1.95966668e-03, 2.11644501e-02],
     [3.75802488e-01, 2.07517079e-03, 0.00000000e+00, 3.72154373e-02, 0.00000000e+00, 5.85927894e-03],
     [4.38037394e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [4.57890626e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [2.75483006e-01, 4.46948936e-03, 0.00000000e+00, 5.29655484e-02, 0.00000000e+00, 1.90997682e-02],
     [4.45203267e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.48695813e-03, 0.00000000e+00],
     [2.92746533e-01, 1.33662475e-02, 1.14261498e-02, 1.05200030e-02, 1.87695369e-01, 9.23965223e-03],
     [3.78274425e-01, 1.43967752e-02, 0.00000000e+00, 9.85239494e-02, 1.35899666e-02, 0.00000000e+00]])

	titles = np.array(['HTTP 404',
 'Alexa Internet',
 'Internet Explorer',
 'HTTP cookie',
 'Google Search',
 'Tumblr',
 'Hypertext Transfer Protocol',
 'Social search',
 'Firefox',
 'LinkedIn',
 'Global warming',
 'Nationally Appropriate Mitigation Action',
 'Nigel Lawson',
 'Connie Hedegaard',
 'Climate change',
 'Kyoto Protocol',
 '350.org',
 'Greenhouse gas emissions by the United States',
 '2010 United Nations Climate Change Conference',
 '2007 United Nations Climate Change Conference',
 'Angelina Jolie',
 'Michael Fassbender',
 'Denzel Washington',
 'Catherine Zeta-Jones',
 'Jessica Biel',
 'Russell Crowe',
 'Mila Kunis',
 'Dakota Fanning',
 'Anne Hathaway',
 'Jennifer Aniston',
 'France national football team',
 'Cristiano Ronaldo',
 'Arsenal F.C.',
 'Radamel Falcao',
 'Zlatan Ibrahimović',
 'Colombia national football team',
 '2014 FIFA World Cup qualification',
 'Football',
 'Neymar',
 'Franck Ribéry',
 'Tonsillitis',
 'Hepatitis B',
 'Doxycycline',
 'Leukemia',
 'Gout',
 'Hepatitis C',
 'Prednisone',
 'Fever',
 'Gabapentin',
 'Lymphoma',
 'Chad Kroeger',
 'Nate Ruess',
 'The Wanted',
 'Stevie Nicks',
 'Arctic Monkeys',
 'Black Sabbath',
 'Skrillex',
 'Red Hot Chili Peppers',
 'Sepsis',
 'Adam Levine'])

	return data, titles

if __name__ == '__main__':
	main()