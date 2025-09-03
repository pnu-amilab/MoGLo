# US3D

**Enhancing Free-hand 3D Photoacoustic and Ultrasound Reconstruction using Deep Learning**  
(link: [https://ieeexplore.ieee.org/document/11036110](https://ieeexplore.ieee.org/document/11036110))

---

## 📁 Project Structure

```
US3D─── data/
    ├── rec/
    └── src/
        ├── utils/
        ├── networks/
        └── options/
```

---

## 🐳 Docker (Strongly Recommended)

This project depends on several complex libraries, including `pytorch3d`, which is notoriously difficult to install manually.

To avoid dependency issues, we **strongly recommend** using the following Docker image:

```bash
docker pull guhong3648/guhong:v1
```

---

## ⚠️ About the Provided Sample Data

- The provided sample dataset is **very limited** and **not suitable** for proper training, validation, or testing.
- It is intended **only for verifying that data loading, model training, and inference pipelines work correctly**.
- We plan to release an **extended dataset** with over **30 subjects** in the near future.

---

## 📂 Public Dataset Used in Our Paper

Also, we used the following public dataset:

> Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu.  
> _"Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames."_  
> In **IEEE ISBI 2023**, pp. 1–5.  
> DOI: [10.1109/ISBI53787.2023.10230773](https://doi.org/10.1109/ISBI53787.2023.10230773)  
> Dataset: [https://zenodo.org/records/7740734](https://zenodo.org/records/7740734)

We used the **central 320×320 region** cropped from each original 480×640 B-mode frame.

For preprocessing scripts and more information, please refer to:  
[https://github.com/ucl-candi/freehand/tree/main](https://github.com/ucl-candi/freehand/tree/main)

---
If you use this code for any comparision experiments, please cite:
```bibtex
@article{lee2025us3d,
  title     = {Enhancing Free-hand 3D Photoacoustic and Ultrasound Reconstruction using Deep Learning},
  author    = {SiYeoul Lee and Seonho Kim and MinKyung Seo and SeongKyu Park and Salehin Imrus and Kambaluru Ashok and DongEon Lee and Chunsu Park and SeonYeong Lee and Jiye Kim and Jae-Heung Yoo and MinWoo Kim},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2025},
  note      = {Early Access},
  doi       = {INSERT_YOUR_DOI}}
```

contact  
For questions or feedback, please contact:
SiYeoul Lee (guhong3648@pusan.ac.kr)
