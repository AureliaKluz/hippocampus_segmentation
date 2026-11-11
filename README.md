# Segmentacja i ocena wolumetryczna hipokampu z danych MRI

Kod do pracy inżynierskiej: segmentacja głowy i trzonu hipokampa na obrazach
MRI (3D U-Net i UNETR, PyTorch/MONAI) na zbiorze Medical Segmentation Decathlon
(Task04_Hippocampus) oraz ocena objętości. Ewaluacja: współczynnik Dice'a,
odległość Hausdorffa (HD, HD95) i błąd objętości.

## Dane
Dane nie są częścią repozytorium. Należy pobrać Task04_Hippocampus ze strony
http://medicaldecathlon.com/ i rozpakować do `data/Task04_Hippocampus/`
(foldery `imagesTr`, `labelsTr`, `imagesTs`). Licencja zbioru: CC-BY-SA 4.0.

## Środowisko
Python 3.12. Należy zainstalować PyTorch zgodnie z instrukcją na pytorch.org
(wersja CPU lub CUDA), a następnie:

    pip install -r requirements.txt

Treningi wykonano na GPU (Tesla T4, Google Colab).

## Kolejność uruchamiania
1. Analiza zbioru i podział danych (powstają pliki w `results/` i `configs/`):

        python src/dataset_summary.py
        python src/check_headers.py
        python src/make_split.py

2. Trening (wyniki w `results/runs/<nazwa>/`: konfiguracja, log, wagi):

        python src/train.py --name baseline --epochs 150 --cache
        python src/train.py --name aug --epochs 150 --cache --augment
        python src/train.py --name unetr --arch unetr --lr 1e-4 --augment --epochs 150 --cache

   Kolejne ziarna losowości: `--seed 43 --name aug_s43` itd.

3. Ewaluacja (zbiór testowy wymaga flagi `--final`):

        python src/evaluate.py --name aug --split val
        python src/evaluate.py --name aug --split test --final

4. Zestawienia, rysunki i pomiary:

        python src/compare_seeds.py --split val
        python src/compare_seeds.py --split test
        python src/error_analysis.py --name aug --split test --final
        python src/make_figures.py
        python src/measure_performance.py

Podział danych (`configs/split.json`) jest stały i nie powinien być zmieniany.
Wytrenowane wagi (`*.pt`) nie są przechowywane w repozytorium.