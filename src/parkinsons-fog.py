# -*- coding: utf-8 -*-
"""599project-tsai.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G2ZUCPP60DGTq3NI07PU5wZcbhONMKO_
"""

!pip install tsai

from tsai.all import *

my_setup()

"""# Data Preprocessing

## Import and Concat Patients
"""

columns = ['timestamp', 'LTA', 'RTA', 'IO', 'ECG', 'RGS', 'accel_x1', 'accel_y1', 'accel_z1', 'gyro_x1', 'gyro_y1', 'gyro_z1', 'NC1', 'accel_x2', 'accel_y2', 'accel_z2', 'gyro_x2', 'gyro_y2', 'gyro_z2', 'NC2', 'accel_x3', 'accel_y3', 'accel_z3', 'gyro_x3', 'gyro_y3', 'gyro_z3', 'NC3', 'accel_x4', 'accel_y4', 'accel_z4', 'gyro_x4', 'gyro_y4', 'gyro_z4', 'SC', 'label']

X_raw=[]
Y_raw=[]
row_count=0
for i in range(1,13):
  if i <= 9:
    task = pd.read_csv(f'/content/00{i}_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp'])
  else:
    task = pd.read_csv(f'/content/0{i}_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp'])
  row_count+=task.shape[0]
  print(row_count)
  if i == 1:
    Y_raw=task.pop('label').to_numpy()
    X_raw=task.to_numpy()
  else:
    Y_raw=np.concatenate((Y_raw,task.pop('label').to_numpy()))
    X_raw=np.concatenate((X_raw,task.to_numpy()))
print(X_raw.shape)
print(Y_raw.shape)

np.mean(Y_raw)

"""## Batch Data"""

BATCH_SIZE=1000
n_batches=len(X_raw)//BATCH_SIZE
# remove remainder rows
X_no_remain=X_raw[:n_batches*BATCH_SIZE,]
Y_no_remain=Y_raw[:n_batches*BATCH_SIZE]

X_batch = np.split(X_no_remain, n_batches)
Y_batch = np.split(Y_no_remain, n_batches)

# n_samples=BATCH_SIZE
# n_features=X.shape[1]
# n_steps=n_batches

# X=np.reshape(X_batch, (n_samples, n_features, n_steps))
# X.shape

n_samples=n_batches
n_features=X_raw.shape[1]
n_steps=BATCH_SIZE

X=np.reshape(X_batch, (n_samples, n_features, n_steps))
X.shape

THRESHOLD=0.5
Y=[]
for i,batch in enumerate(Y_batch):
  batch_mean=np.mean(batch)
  if batch_mean > THRESHOLD:
    Y.append(1)
  else:
    Y.append(0)
Y=np.array(Y)
Y.shape

Y_batch[93]

"""# Begin TSAI Setup and Training"""

tfms = [None, [Categorize()]]
dsets = TSDatasets(X, Y, tfms=tfms, inplace=True)

dsets.valid

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, batch_tfms=[TSStandardize()], num_workers=0)

dls.show_batch(sharey=True)

# # Build Learner
# c_in=n_features
# c_out=1
# model = LSTM(dls.vars, dls.c)
# model = LSTM (c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0,
#        bidirectional=False, fc_dropout=0.0)
model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
learn.save('stage0')

learn.load('stage0')
learn.lr_find()

learn.fit_one_cycle(1, lr_max=1e-3)
learn.save('stage1')

learn.recorder.plot_metrics()

learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

batch_tfms = TSStandardize(by_sample=True)
mv_clf = TSClassifier(X, Y, path='models', arch=LSTM, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
mv_clf.fit_one_cycle(10, 1e-2)
mv_clf.export("mv_clf.pkl")

"""## Import Data"""

columns = ['timestamp', 'LTA', 'RTA', 'IO', 'ECG', 'RGS', 'accel_x1', 'accel_y1', 'accel_z1', 'gyro_x1', 'gyro_y1', 'gyro_z1', 'NC1', 'accel_x2', 'accel_y2', 'accel_z2', 'gyro_x2', 'gyro_y2', 'gyro_z2', 'NC2', 'accel_x3', 'accel_y3', 'accel_z3', 'gyro_x3', 'gyro_y3', 'gyro_z3', 'NC3', 'accel_x4', 'accel_y4', 'accel_z4', 'gyro_x4', 'gyro_y4', 'gyro_z4', 'SC', 'label']

task1=pd.read_csv('/content/001_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y1=task1.pop('label').to_numpy()
X1=task1.to_numpy()

task2=pd.read_csv('/content/002_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y2=task2.pop('label').to_numpy()
X2=task2.to_numpy()

task3=pd.read_csv('/content/003_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y3=task3.pop('label').to_numpy()
X3=task3.to_numpy()

task4=pd.read_csv('/content/004_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y4=task4.pop('label').to_numpy()
X4=task4.to_numpy()

task5=pd.read_csv('/content/005_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y5=task5.pop('label').to_numpy()
X5=task5.to_numpy()

task6=pd.read_csv('/content/006_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y6=task6.pop('label').to_numpy()
X6=task6.to_numpy()

task7=pd.read_csv('/content/007_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y7=task7.pop('label').to_numpy()
X7=task7.to_numpy()

task8=pd.read_csv('/content/008_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y8=task8.pop('label').to_numpy()
X8=task8.to_numpy()

task9=pd.read_csv('/content/009_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y9=task9.pop('label').to_numpy()
X9=task9.to_numpy()

task10=pd.read_csv('/content/010_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y10=task10.pop('label').to_numpy()
X10=task10.to_numpy()

task11=pd.read_csv('/content/011_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y11=task11.pop('label').to_numpy()
X11=task11.to_numpy()

task12=pd.read_csv('/content/012_task_1.txt', delimiter=',', names=columns).drop(columns=['timestamp']).head(94000)
Y12=task12.pop('label').to_numpy()
X12=task12.to_numpy()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

task1

changes=np.where(Y1[:-1] != Y1[1:])[0]
changes

# plt.figure(figsize=(18, 6))
fig, ax = plt.subplots(figsize=(18, 6))
ax.set_title('Patient 1 - Task 1')

step=750
columns = ['LTA', 'RTA', 'IO', 'ECG', 'RGS', 'accel_x1', 'accel_y1', 'accel_z1', 'gyro_x1', 'gyro_y1', 'gyro_z1', 'NC1', 'accel_x2', 'accel_y2', 'accel_z2', 'gyro_x2', 'gyro_y2', 'gyro_z2', 'NC2', 'accel_x3', 'accel_y3', 'accel_z3', 'gyro_x3', 'gyro_y3', 'gyro_z3', 'NC3', 'accel_x4', 'accel_y4', 'accel_z4', 'gyro_x4', 'gyro_y4', 'gyro_z4', 'SC']
for col in columns:
  ax.plot(task1[col][::step])
for i,loc in enumerate(changes):
  if (i % 2) == 0:
    rect=patches.Rectangle((loc,-30000),changes[i+1]-loc,60000,alpha=0.5)
    ax.add_patch(rect)

FX=np.array([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12])
y=np.array([Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10,Y11,Y12])

X=np.array([X1,X2,X3,X4,X5,X6])
y=np.array([Y1,Y2,Y3,Y4,Y5,Y6])

X.shape, y.shape

# X = X[:, 0]
# y = y.reshape(-1, 1)
# data = np.concatenate((X, y), axis=-1)
# df = pd.DataFrame(data)
# df.head()

# X, y = df2xy(df, target_col='target')
# test_eq(X.shape, (60, 1, 570))
# test_eq(y.shape, (60, ))

# splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
# splits



# X, y = df2xy(df, target_col='target')
# print(X.shape)
# test_eq(X.shape, (60, 1, 570))
# test_eq(y.shape, (60, ))

# X_memmap=np.memmap(self._prepared_data_location_npmemmap_X,dtype='float32',mode='w+')
# X_list_total_standardized_memmap[:]=X_list_total_standardized[:]

"""##Train"""

# X, y, splits = get_classification_data('LSST', split_data=False)
batch_tfms = TSStandardize(by_sample=True)
mv_clf = TSClassifier(X, y, path='models', arch=InceptionTimePlus, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
mv_clf.fit_one_cycle(10, 1e-2)
mv_clf.export("mv_clf.pkl")