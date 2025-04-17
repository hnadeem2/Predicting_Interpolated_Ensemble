import numpy as np
import mdtraj as md
import pickle 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm.notebook import tqdm
from deeptime.util import energy2d
from deeptime.decomposition import TICA
from deeptime.plots import plot_energy2d
import sys
import glob
import natsort
import re

purple=tuple(['#6247aa','#815ac0','#a06cd5','#b185db','#d2b7e5'])
blue=tuple(['#2c7da0','#468faf','#61a5c2','#89c2d9','#a9d6e5'])
green=tuple(['#718355', '#87986a', '#97a97c', '#a3b18a', '#cfe1b9'])
orange=tuple(['#ffb700','#ffc300','#ffd000','#ffdd00','#ffea00'])
red=tuple(['#f25c54', '#f27059', '#f4845f', '#f79d65', '#f7b267'])
larger=tuple(['#f7b267','#f7b267','#f7b267','#f7b267','#f7b267'])
anh_colors = purple+blue+green+orange+red+larger
anh_cmap = ListedColormap(anh_colors)

def fep(x, y,weights=None,*args, x_label='RMSD (Å)', y_label='g (Å)',cmap=('nipy_spectral'),save=False,name='mono'):
   

    '''
        Plots 2D Free Energy plot for x, y
    
    '''
    
    energies = energy2d(x, y, bins=(200, 200), kbt=2.479/4.184, weights=weights, shift_energy=True)
    z1lim = [0, 5]
    levels=np.linspace(*z1lim,30)
    #vmax=6
    ax, contour, cbar = energies.plot(contourf_kws=dict(cmap=anh_cmap),levels=levels)

    ticks = np.linspace(*z1lim, 6)

    # cbar.ax.tick_params(axis='y',labelsize=14)
    # cbar.ax.set_yticks(ticks)
    # cbar.ax.set_yticklabels(ticks)
    # cbar.set_label('Free Energy (kcal/mol)',fontsize=14)
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel(x_label,fontsize=14)
    ax.set_ylabel(y_label,fontsize=14)
    cbar.remove()
   # ax.grid()
    
    if save:
        plt.savefig(f"fep_{name}.png", dpi=300)
    
    return energies

feats = pickle.load(open("PfHT1_apo_features_clean.pkl","rb"))
# #projection = pickle.load(open(f'tic_projection.pkl','rb'))
# # tica1, tica2 = np.concatenate(projection)[:,0], np.concatenate(projection)[:,1]
# dis1, dis2 = np.concatenate(feats)[:,14], np.concatenate(feats)[:,13]


# infc = md.load("IF.pdb")
# outfc = md.load("OF.pdb")
# # infc_model = md.load("IF_model_0.pdb")
# # outfc_model = md.load("OF_model_0.pdb")
# gen_pdbs =  natsort.natsorted(glob.glob("aggregated_pdbs*/*.pdb"))

# def return_dis(pdb_list):

# 	x_coord = []
# 	y_coord = []
# 	for pdb in pdb_list:
# 		traj = md.load(pdb)
# 		l47_v314 = traj.top.select('name CA and (resid 46 or resid 313)').reshape(1,-1)
# 		I152_H416 = traj.top.select('name CA and (resid 151 or resid 415)').reshape(1,-1)
# 		y_dis = md.compute_distances(traj,l47_v314)[0]
# 		x_dis = md.compute_distances(traj,I152_H416)[0]
# 		x_coord.append(x_dis)
# 		y_coord.append(y_dis)
# 	return np.concatenate(x_coord), np.concatenate(y_coord)

# if_x, if_y = return_dis(["IF.pdb"])
# of_x, of_y = return_dis(["OF.pdb"])
# gen_x, gen_y = return_dis(gen_pdbs)



# fep(dis1,dis2,y_label="dis-2",x_label="dis-1")#,weights=weights)
# plt.scatter(if_x,if_y,marker='o',color='black',label="IF")
# plt.scatter(of_x,of_y,marker='o',color='red',label="OF")

# print("Plotting generated points")
# for x,y,text in zip(gen_x,gen_y,gen_pdbs):
#     # print(text)
#     # sys.exit()
# 	# lambda_param = float(re.search(r'lambda([0-9.]+)', text).group(1))
# 	# plt.annotate(lambda_param,xy=(x,y),fontsize=4)
# 	plt.scatter(x,y,marker='o', color = 'orange')

    
# # plt.scatter(gen_x_A,gen_y_A,marker='x', color = 'black', label ="intp IF")
# # plt.scatter(gen_x_B,gen_y_B,marker='x', color = 'red', label ="intp B")
# # plt.scatter(if_model_x,if_model_y,marker='x', color = 'yellow', label ="IF model")
# # plt.scatter(of_model_x,of_model_y,marker='x', color = 'orange', label ="OF model")

# plt.tight_layout()
# plt.xlim(0.6,1.6)
# plt.ylim(0.6,2.2)
# plt.legend()
# plt.savefig(f'dis_orig.jpg',dpi=400)
# #plt.show()

######################################################################################

gen_pdbs =  natsort.natsorted(glob.glob("aggregated_pdbs*/*.pdb"))



def get_angle(t):
    Atom1 = md.compute_center_of_mass(t, select=("resid 306 to 308")) 
    Atom2 = md.compute_center_of_mass(t, select=("resid 313 to 315")) 
    Atom3 = md.compute_center_of_mass(t, select=("resid 321 to 323")) 
    Vec12 = (Atom1 - Atom2)
    Vec23 = (Atom3 - Atom2)
    cosangle = []
    for i in range(t.n_frames):
        tempcosangle = np.dot(Vec12[i],Vec23[i])/(np.linalg.norm(Vec12[i])*np.linalg.norm(Vec23[i]))
        cosangle.append(tempcosangle)

    angle = []
    for i in range(t.n_frames):
        tempangle = np.arccos(cosangle[i])*180/np.pi
        angle.append(tempangle)

    # Convert angle to a NumPy array
    angle_array = np.array(angle)

    return angle_array


def return_dis(pdb_list):
    dis_list = []
    angle_list = []
    for pdb in pdb_list:
        traj = md.load(pdb)
        l47_v314 = traj.top.select('name CA and (resid 46 or resid 313)').reshape(1,-1)
        I152_H416 = traj.top.select('name CA and (resid 151 or resid 415)').reshape(1,-1)
        y_dis = md.compute_distances(traj,l47_v314)[0]
        x_dis = md.compute_distances(traj,I152_H416)[0]
        dis = y_dis - x_dis
        angle = get_angle(traj)
        dis_list.append(dis)
        angle_list.append(angle)
    return dis_list, angle_list



if_dis, if_angle = return_dis(["IF.pdb"])
of_dis, of_angle = return_dis(["OF.pdb"])
gen_dis, gen_angle = return_dis(gen_pdbs)


angle, dis1, dis2 = np.concatenate(feats)[:,15], np.concatenate(feats)[:,14], np.concatenate(feats)[:,13] 
fep(dis2-dis1,angle,y_label="angle",x_label="dis2-dis1")#,weights=weights)

# print("Plotting generated points")
# for x,y,text in zip(gen_dis,gen_angle,gen_pdbs):
    
#     lambda_param = float(re.search(r'aggregated_pdbs_([01](?:\.\d+)?)', path).group(1))
#     #plt.annotate(lambda_param,xy=(x,y),fontsize=4)
#     plt.scatter(x,y,marker='o', color = 'orange')



cmap = plt.cm.plasma
for x, y, path in zip(gen_dis, gen_angle, gen_pdbs):
    lam = float(re.search(r'aggregated_pdbs_([01](?:\.\d+)?)', path).group(1))
    plt.scatter(x, y, color=cmap(lam), marker='o')
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='lambda')


plt.scatter(if_dis,if_angle,marker='o',color='black',label="IF")
plt.scatter(of_dis,of_angle,marker='o',color='red',label="OF")
plt.xlim(-0.8,1.2)
plt.ylim(75,180)
plt.savefig(f'angle.jpg',dpi=400)