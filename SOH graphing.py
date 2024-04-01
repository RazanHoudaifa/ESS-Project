import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import glob
import seaborn as sns

dir = './content/'
listdir = os.listdir(dir)

print(listdir)
print("The number of dataset :", len(listdir))

num = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
for i in range(len(num)):
    vector = np.zeros((1,3))
    path = os.path.join(os.getcwd(), './content/', num[i] + '_discharge_soh.csv')
    csv = pd.read_csv(path)
    df = pd.DataFrame(csv)
    
    vec = df[['cycle', 'capacity', 'SOH']]
    
    # Store vec in the global namespace
    globals()['data_{}'.format(num[i])] = vec

for i in range(len(num)):
    print("Shape of data :", np.shape(globals()['data_{}'.format(num[i])]))

data = pd.read_csv('./content//B05_discharge_soh.csv')
df = pd.DataFrame(data)
df
for i in range(len(num)):
    print("Shape of data :", np.shape(globals()['data_{}'.format(num[i])]))

data_B05

for i in range(len(listdir)) :

    dff = globals()['data_{}'.format(num[i])]
    
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))

    plt.scatter(dff['cycle'], dff['SOH'])
#     plt.plot(dff['cycle'], len(dff['cycle'])*[0.7], color = 'red')

    plt.ylabel('SoH', fontsize = 15)
    plt.xlabel('cycle', fontsize = 15)
    plt.title('Discharge_' + num[i], fontsize = 15)
    plt.savefig('./content/_PICS1' + num[i] + '.jpg')
    plt.show()




# Group A

sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))

plt.scatter(data_B05['cycle'], data_B05['SOH'],label='B05')
plt.scatter(data_B07['cycle'], data_B07['SOH'],label='B07')
plt.scatter(data_B18['cycle'], data_B18['SOH'],label='B18')

plt.legend(prop={'size': 16})

plt.ylabel('SoH', fontsize = 15)
plt.xlabel('Discharge cycle', fontsize = 15)
plt.title('SoH of group A', fontsize = 15)
plt.savefig('./content/_PICS1/A_group.jpg')
plt.show()



# Group B

sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))

plt.scatter(data_B33['cycle'], data_B33['SOH'],label='B33')
plt.scatter(data_B34['cycle'], data_B34['SOH'],label='B34')

plt.legend(prop={'size': 16})

plt.ylabel('SoH', fontsize = 15)
plt.xlabel('Discharge cycle', fontsize = 15)
plt.title('SoH of group B', fontsize = 15)
plt.savefig('./content/_PICS1/B_group.jpg')
plt.show()



# Group C

sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))

plt.scatter(data_B46['cycle'], data_B46['SOH'],label='B46')
plt.scatter(data_B47['cycle'], data_B47['SOH'],label='B47')
plt.scatter(data_B48['cycle'], data_B48['SOH'],label='B48')

plt.legend(prop={'size': 16})

plt.ylabel('SoH', fontsize = 15)
plt.xlabel('Discharge cycle', fontsize = 15)
plt.title('SoH of group C', fontsize = 15)
plt.savefig('./content/_PICS1/C_group.jpg')
plt.show()
