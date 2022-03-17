# BiLSTM_CRF_NER

#### Description
This project serves as the final assignment of the first semester of 2021-2022 comprehensive course practice of artificial intelligence in the School of Computer and Artificial Intelligence of Wuhan University of Technology, and its main part refers to the web open source code.

#### Developing Environment
This project is developed under Ubuntu 20.04 operating system, the main environment configuration is: Python3.8, PyTorch1.7, CUDA11.3, NVIDIA GeForce MX110, etc.

#### Installation

1.  Pull the repo to a local dir.
2.  Install essential packages for this project.
3.  Execute the specified command using the command line.

#### Structure
D.
│  .gitignore
│  predict.py
│  predict_input_data.txt
│  train.py
│          
├─data
│  │  Bosondata.pkl
│  │  dataMSRA.pkl
│  │  renmindata.pkl
│  │  yidu-s4k.pkl
│  │  
│  ├─boson
│  │      
│  ├─MSRA
│  │      
│  ├─renMinRiBao
│  │      
│  └─yidu-s4k
│          
├─model
│      
├─res
│      
└─utils
    │  BiLSTM_CRF.py
    │  resultCal.py
    │  str2bool.py
    └─ __init__.py
