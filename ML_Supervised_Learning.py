"""
ML(Machine Learning)
Supervised Learning (Denetimli Öğrenme= Feature(+) + Label(+))
Modelin hem giriş verileri(x)
hem de bu verilere ait doğruları cevapları (Etiketleri[Label]: Y)

Temel Yapı:
    X(Features/Özellikler) --> MODEL --> y (Label/Etiketler)

Örneğin:
- E-Posta spam mı değil mi ?
- Ev fiyatları ne kadar olur ?
- Bu müşteriye borç beyaz eşya verilir mi ?

Öğrenme türü olan LABEL vardır.
"""

"""
Bir öğrencinin:
1-) Günlük çalışma saati
2-) Derse katılım yüzdesi
bu bilgilere bakarak sınavı geçip geçmeyeceğini tahmin edelim.

Label:
    0 : Kaldı
    1 : Geçti
    
Kullanılan algoritma:
    Logistic Regression  
    
    
Kurulum:
    pip install numpy scikit-learn
    python -c "import numpy; import sklearn; print('Kurulum başarılı')"
    
    python -m pip install -r requirements.txt
    
    -m → module
    -c → command
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    #  X(Features/Özellikler) --> MODEL --> y (Label/Etiketler)
    #  1= Günlük çalışma saati
    #  30= Derse katılım yüzdesi
    X = np.array([
        [1, 30],
        [2, 40],
        [2, 50],
        [3, 55],
        [4, 60],
        [5, 65],
        [6, 75],
        [7, 85],
        [8, 90],
        [9, 95]
    ])

    # y (Label/Etiketler)
    # 0 = Kaldı, 1= Geçti

    # NOT: Supervised Learning'in en önemli özelliği ==>
    # X verileriyle birlikte y etiketlerinin bulunmasıdır(SL)

    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    print("=== SUPERVISED LEARNING (Features(+) Label(+))")
    print("\nX - Öğrenci Özellikler(Features)")
    print(X)

    print("\ny - Label(Etiketler)")
    print(y)
