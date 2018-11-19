# -*- coding:utf-8 -*-


from __future__ import division
from gensim.models import word2vec
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import numpy as np
import xlrd


"""
optional:
if there is a error in the encode('utf-8'),
should set the sys encoding to 'utf-8',

"""
reload( sys )
sys.setdefaultencoding( 'utf8' )


data_path = os.path.abspath( '.' )
xlsx_file = xlrd.open_workbook( data_path + '/oldFile_hitProducts.xlsx')
model = word2vec.Word2Vec.load( data_path + "/cookpad_v2.model" )


sheet_name = xlsx_file.sheet_names()[0]
sheet = xlsx_file.sheet_by_index( 0 )


#calculate distance of two items, if the i = j, its means the same item, so the distance is 0
dist_space = []


"""
get the length of ni * nj for average distance calculation
because the xlrd package, each row has the same columns with the max columns item

"""
lenth = []
for ith in range( 1, 87 ):

    count = 0

    for col in range( 3, len(sheet.row(1) )):


        word = sheet.row( ith )[col].value
        word.encode('utf-8')


        if len( word ) <= 0:

            break

        else:

            count += 1

    lenth.append( count )

lenth = np.array( lenth )


for i_th in range( 1, 87 ):

    elem = []

    for j_th in range( 1, 87 ):

        temp_sum = 0.0


        for col in range( 3, len(sheet.row(1)) ):

            wordi = sheet.row( i_th )[col].value
            wordi.encode( 'utf-8' )

            wordj = sheet.row( j_th )[col].value
            wordj.encode( 'utf-8' )

            if len( wordi ) <= 0 or len( wordj ) <= 0:

                break

            else:
                Wi = np.array( model[wordi] ).reshape((1, 200))
                Wj = np.array( model[wordj] ).reshape((1, 200))

                temp_dist = np.sum( pow( (Wi - Wj), 2), axis=1, keepdims=True )
                temp_sum += np.sqrt( temp_dist )

        temp_sum = temp_sum / ( lenth[i_th-1] * lenth[j_th-1] )
        elem.append( temp_sum )

    dist_space.append( elem )

mds = TSNE( n_components=3, init='pca', random_state=0 )

dist_matrix = np.array( dist_space ).reshape((86, 86))
print dist_matrix, dist_matrix.shape
#print dist_matrix[14].reshape((1, 86))
#print dist_matrix[13].reshape((1, 86))
#assert( dist_matrix.shape == ( 86, 86 ) )
changed = mds.fit_transform( dist_matrix )
print changed, changed.shape

np.savetxt( "3dims_pos.csv", changed, delimiter="," )


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = [ changed[i][0] for i in range( 86 ) ]
ys = [ changed[i][1] for i in range( 86 ) ]
zs = [ changed[i][2] for i in range( 86 ) ]

ax.scatter(xs, ys, zs, c='r', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#plot the items after be dealt by MDS_2dim
"""
similarities = euclidean_distances( dist_matrix )
s = 100
fig = plt.figure( 1 )
ax = plt.axes([0., 0., 1., 1.])

plt.scatter( changed[:, 0], changed[:, 1], color='turquoise', s=s, lw=0, label='MDS')
plt.legend( scatterpoints=1, loc='best', shadow=False )

similarities = similarities.max() / similarities * 100
similarities[np.isinf(similarities)] = 0


start_idx, end_idx = np.where( changed )
segments = [[changed[i, :], changed[j, :]]
            for i in range(86) for j in range(86)]
values = np.abs( similarities )
lc = LineCollection(segments, zorder=0, cmap=plt.cm.Blues, norm=plt.Normalize(0,values.max()))
lc.set_array( similarities.flatten())
lc.set_linewidths( np.full(len(segments), 0.5))
ax.add_collection( lc )

plt.show()
"""
