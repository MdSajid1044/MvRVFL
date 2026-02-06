# MvRVFL: Multiview Random Vector Functional Link Network for Predicting  DNA-Binding Proteins

Please cite the following paper if you are using this code. Reference: A. Quadir, M. Sajid, and M. Tanveer. "Multiview random vector functional link network for predicting DNA-binding proteins." arXiv preprint arXiv:2409.02588 (2026).

BibTex
-------
```
@article{quadir2024multiview,
  title={Multiview random vector functional link network for predicting DNA-binding proteins},
  author={Quadir, Abdul and Sajid, M and Tanveer, M},
  journal={arXiv preprint arXiv:2409.02588},
  year={2024}
}
```

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
The experiments are executed on a computing system running Python 3.11, an Intel(R) Xeon(R) CPU E5-2697 v4 processor operating at 2.30 GHz with 128 GB of Random Access Memory (RAM), and Windows 11.

We have put a demo of the “MvRVFL” model with the “breast_cancer_wisc_prog” dataset

In this demonstration, the codes are run for a fixed hyperparameter for the “breast_cancer_wisc_prog” dataset
c1 = 0.00001
c2 = 1000
rho =0.01
N = 23

For detailed hyperparameter settings, please refer to the paper.

Description of files: 
main.py: This is the main file to run selected algorithms on datasets. In the path variable, specify the path to the folder containing the codes and datasets on which you wish to run the algorithm. 

RVFL.py: Solving the optimization problem.

For the detailed experimental setup, please follow the paper. If you find any bugs/issues, please write to A. Quadir (mscphd2207141002@iiti.ac.in) and M. Sajid (phd2101241003@iiti.ac.in).


