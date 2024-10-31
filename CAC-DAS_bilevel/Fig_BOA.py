import numpy as np
import matplotlib.pyplot as plt


root = "./results_old/results_0/home/ubuntu/V22_instance/CAC-DAS_bilevel/Tune_T=500_R=100/"
path = root+"Bilevel_CACm_online_obj_AIRBUS_w_0.5_0.0_0.0_0.5_a_4.0_5.0__"

Hkeys = ["Hbest","Hmin"]#,"obj_best","Havg"]
Hkeys_label = [r"$H^*$",r"min $H$"]#,r"$Q$",r"$<H>$"]
        
xkeys = ["lamb1","lamb2","beta","a","gamma","xi","beta_BP"]
xkeys_label = [r"$\lambda_1$",r"$\lambda_2$",r"$\beta$(CAC)",r"$a$",r"$\gamma$",r"$\xi$",r"$\beta$(IBP)"]
           
# Fig (a) energy and objective

plt.figure()

ax1 = plt.subplot(2,1,1)

for key, key_label in zip(Hkeys, Hkeys_label):
    full_path = f"{path}{key}.txt"  # Assuming the files are in .txt format

    data = np.loadtxt(full_path)
    steps = range(1,len(data))
    
    ax1.plot(steps, data[1:], label=key_label)  # Adjust x and y as needed depending on your data
    
ax1.set_xlabel(r'DAS step $n$')
ax1.set_ylabel(r'$H$ (Ising)')

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

key = "obj_best"
key_label = r"$Q$"
full_path = f"{path}{key}.txt"  # Assuming the files are in .txt format
data = np.loadtxt(full_path)
steps = range(1,len(data))
ax2.plot(steps, data[1:],'--r', label=key_label)  # Adjust x and y as needed depending on your data
    
ax2.set_xlabel(r'$n$ (DAS step )')
ax2.set_ylabel(r'$Q$ (QUBO)')

ax1.set_xlim(1,len(data))
ax2.set_xlim(1,len(data))

# Merge legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left",ncols=2,fontsize=8, bbox_to_anchor=(0.16, 0.5), handlelength=1, handletextpad=0.4, labelspacing=0.2)  # Adjust location as needed

ax1.text(-0.2, 1.0, "a", transform=ax1.transAxes, fontsize=12, va='top', ha='right')

# Fig (b) hyperparameters

ax3 = plt.subplot(2,1,2)

for key, key_label in zip(xkeys, xkeys_label):
    full_path = f"{path}{key}.txt"  # Assuming the files are in .txt format

    data = np.loadtxt(full_path)
    steps = range(0,len(data))
    
    plt.plot(steps, data, label=key_label)  # Adjust x and y as needed depending on your data
    
plt.xlabel(r'$n$ (DAS step )')
plt.ylabel(r'$\theta$ (Solver parameters )')

plt.xlim(1,len(data))

plt.legend(ncols=3,fontsize=8,bbox_to_anchor=(0.52, 0.42), handlelength=1, handletextpad=0.4, labelspacing=0.2)
ax3.text(-0.2, -0.5, "b", transform=ax1.transAxes, fontsize=12, va='top', ha='right')

plt.tight_layout()

plt.savefig("./results/BOA.eps", format="eps", dpi=300)  # Adjust dpi as needed
plt.savefig("./results/BOA.png", format="png", dpi=300)  # Adjust dpi as needed