# CCHNET

by Meiyi Yang, Yang Lili, Zhang Qi, Lifeng Xu, Bo Yang, Yingjie Li, Xudong Cheng, Feng Zhang, Minghui Liu, Tianshu Xie, Xuan Cheng2, Ming Liu1, Nengwei Yu

## Abstract

	Chronic cerebral hypoperfusion (CCH) is a state of blood supply reduction to the brain, which has been shown to play a significant role in Alzheimer's disease, Vascular dementia, and other nervous disease. However, effective diagnostic tools for CCH are currently limited during routine examinations. This study aimed to develop and evaluate an MRI-based deep learning algorithm for detecting CCH, promoting a more unified, standardized, and scientific clinical treatment approach. In the retrospective research, we collected 204 routine brain magnetic resonance imaging (MRI) from March 1 to September 10, 2022, as training and testing cohorts. MRI sequences were processed to obtain T1-weighted (T1WI) and T2-weighted (T2WI) sequence images for each patient. We propose CCHNet, an efficient combination of transformers and discriminators, designed to automatically extract persuasive features for CCH prediction using these images. To better validate model performance, a validation cohort with 21 samples was collected from November 14, 2022, to April 10, 2023. Four neurologists with different professional levels independently performed diagnosis within the validation cohort. The diagnostic performance of the proposed model and the neurologists' assessment were evaluated and compared. The proposed model achieved an 85.4% accuracy, 83.0% sensitivity, 90.0% specificity, and 91.6% area under the curve (AUC) in the testing cohort. Notably, the model's accuracy for diagnosing CCH at 85.7% was significantly higher than that of senior neurologists at 61.5%. The CCHNet-assisted strategy also improved the pooled AUC of senior neurologists from 57.5% to 87.5% in the validation cohort. Additionally, we identified correlations between CCH and factors such as headache, diabetes, smoking, and drinking, with Pearson correlation coefficients of 0.058, 0.187, 0.144, and 0.187, respectively.

<img src="C:\Users\yangmeiyi\Desktop\CCH_Diagnosis\network.png" style="zoom:38%;" />

## Code

### Recommended environment

- Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
- Python >= 3.9.7
- PyTorch >= 1.13.0
- timm >=0.6.11
- torchvision >=0.14.0
- cuda 11.4

### Usage

python3   ./CCHNet/train.py

### Data Preparation

1. Preprocess the data. The default model takes images of size 224x 224.

2. The data files include train.csv test.csv, and val.csv, with formats including

   ```
   { "name": patientID, "cch": label}
   ```

   

3. Create a patient dictionary. This should be a pickle file containing a dict as follows, where img is the image matrix of slice :

   ```
   dataset = {
       "img": img_data,
       "label": torch.Tensor([y])[0],
       "name": patientID,
   }
   ```





